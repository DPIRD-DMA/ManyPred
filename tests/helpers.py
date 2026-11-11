"""Shared test helpers: small deterministic models, devices, timeouts."""

from __future__ import annotations

import threading
import time
import traceback

import numpy as np
import pytest
import torch

IMAGE_SIZE = 64

DEVICES = [
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        ],
    ),
    pytest.param(
        "mps",
        marks=[
            pytest.mark.mps,
            pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"),
        ],
    ),
]
ALL_DEVICES = [pytest.param("cpu"), *DEVICES]


class SeededConv(torch.nn.Module):
    """3 -> 4 channel 3x3 conv with seeded weights."""

    def __init__(self, in_channels: int = 3, classes: int = 4) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(in_channels, classes, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=generator))
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FailOnCall(SeededConv):
    """Raises on the given call."""

    def __init__(self, fail_on: int = 2) -> None:
        super().__init__()
        self.calls = 0
        self.fail_on = fail_on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("model failed")
        return super().forward(x)


def random_images(count: int = 5, size: int = IMAGE_SIZE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((count, 3, size, size)).astype(np.float32) + 0.01


def striped_images(count: int, stripe: int = 4) -> np.ndarray:
    """Random images where image i has a nodata stripe at rows i*stripe, so each masked
    output identifies which input it came from."""
    data = random_images(count, seed=3) + 0.1
    for i in range(count):
        data[i, :, i * stripe : (i + 1) * stripe, :] = 0
    return data


def run_with_timeout(func, timeout: float = 120):
    """Run func in a daemon thread, failing the test if it hangs, else returning its result
    or re-raising its error."""
    outcome = {}

    def target():
        try:
            outcome["result"] = func()
        except BaseException as e:
            # release the frames' locals as a real caller would, so open generators close
            traceback.clear_frames(e.__traceback__)
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"Call did not finish within {timeout}s, it may have deadlocked")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def wait_for_new_threads(baseline: set, timeout: float = 5) -> list:
    deadline = time.monotonic() + timeout
    while True:
        new = [t for t in threading.enumerate() if t not in baseline and t.is_alive()]
        if not new or time.monotonic() > deadline:
            return new
        time.sleep(0.05)
