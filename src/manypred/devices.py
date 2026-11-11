"""Devices, dtypes, the MPS lock and MPS workarounds."""

from __future__ import annotations

import copyreg
import os
import sys
import warnings
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from threading import RLock
from typing import Any

import torch

# PyTorch's MPS backend shares one command buffer per process, so GPU work encoded from two
# threads at once fails a Metal assertion and aborts the process.
_MPS_LOCK = RLock()

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
}


def warn_caller(message: str) -> None:
    """Warn at the first frame outside ManyPred (the user's call)."""
    package = os.path.dirname(os.path.abspath(__file__)) + os.sep
    frame, level = sys._getframe(1), 2  # stacklevel 2 is warn_caller's caller
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(package):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, UserWarning, stacklevel=level)


def default_device() -> torch.device:
    """CUDA if available, else MPS, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """A torch.device from a name, a device, or None (:func:`default_device`).

    ``"cuda"`` without an index becomes the caller's current GPU (``"cuda:N"``). The current
    GPU is set per thread (``torch.cuda.set_device``), so the prefetch and saving threads,
    which start on GPU 0, have to be told which one the caller meant."""
    resolved = default_device() if device is None else torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def same_device(a: torch.device, b: torch.device) -> bool:
    """Whether two devices are the same, counting a device without an index (``"cuda"``) as
    the current one of its type (``"cuda:0"``)."""
    if a.type != b.type:
        return False
    if a.index is None or b.index is None:
        current = torch.cuda.current_device() if a.type == "cuda" else 0
        return (a.index if a.index is not None else current) == (
            b.index if b.index is not None else current
        )
    return a.index == b.index


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """A floating torch dtype from a name (``"float32"``, ``"fp16"``, ``"bf16"``, ...) or a
    torch dtype."""
    if isinstance(dtype, torch.dtype):
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating point type, got {dtype}.")
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower().removeprefix("torch.")
        if key in _DTYPES:
            return _DTYPES[key]
        raise ValueError(f"Unknown dtype {dtype!r}. Use one of {sorted(_DTYPES)}.")
    raise TypeError(f"dtype must be a str or torch.dtype, got {type(dtype).__name__}.")


def inference_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    """The dtype to run in, with a warning for reduced precision on the CPU."""
    resolved = resolve_dtype(dtype)
    warn_cpu_reduced_precision(device, resolved)
    return resolved


def device_lock(*devices: torch.device) -> AbstractContextManager:
    """Serialise work across threads when any of the devices is MPS. Other devices don't
    need it."""
    if any(device.type == "mps" for device in devices):
        return _MPS_LOCK
    return nullcontext()


def warn_cpu_reduced_precision(device: torch.device, dtype: torch.dtype) -> None:
    """Warn that fp16/bf16 inference on the CPU is usually much slower than fp32."""
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        warn_caller(
            f"Inference on the CPU in {dtype} is usually much slower than in float32. "
            f"Pass dtype='float32' for CPU inference."
        )


class MPSSafeConv2d(torch.nn.Conv2d):
    """Conv2d that makes its input contiguous on MPS.

    Some models (e.g. timm EdgeNeXt) feed non-contiguous tensors to convolutions, which MPS
    rejects with "view size is not compatible with input tensor's size and stride".

    It pickles (and deep-copies) as a plain ``torch.nn.Conv2d``, so a model saved with
    ``torch.save(model)`` after an MPS run loads without ManyPred.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type == "mps" and not input.is_contiguous():
            input = input.contiguous()
        return super().forward(input)

    def __reduce_ex__(self, protocol: Any) -> Any:
        # rebuilt as a plain Conv2d (pickle's __newobj__ would insist on this class)
        state = super().__reduce_ex__(protocol)[2]
        return copyreg._reconstructor, (torch.nn.Conv2d, object, None), state


def patch_for_mps(models: Iterable[object]) -> None:
    """Make every Conv2d in the models MPS-safe, in place (:class:`MPSSafeConv2d`).

    The swap changes each plain ``torch.nn.Conv2d`` module's class and stays after the run
    (the models may run again, on other threads), so ``type(module) is torch.nn.Conv2d``
    becomes False; ``isinstance`` checks, ``state_dict()`` and pickling are unaffected. Models
    are patched whenever the device is MPS, as the patched forward only adds a contiguity
    check.
    """
    for model in models:
        if not isinstance(model, torch.nn.Module):
            continue
        for module in model.modules():
            if type(module) is torch.nn.Conv2d:
                module.__class__ = MPSSafeConv2d
