"""Golden files: the tiny test models, case inputs and loaders.

Shared by ``scripts/make_goldens.py`` (which runs OmniCloudMask to write the files) and the
parity tests (which run ManyPred on the same inputs and compare). Golden files live in
``tests/golden/``:

* ``inputs.npz``: the base crops (``s2``: (3, 600, 760) uint16 Sentinel-2 L2A, R/G/NIR at
  10 m, cloudy; ``lc8``: (3, 512, 640) uint16 Landsat 8 L1, a scene corner with ~40% nodata
  along a diagonal edge) and their georeferencing as JSON.
* ``models.npz``: weights of the seeded tiny models (``tiny0``, ``tiny1``).
* ``cases/<name>.npz``: one OCM run each, with its parameters as JSON (``meta``), the output
  (``output``) and, for class outputs, the flat indexes of pixels whose top two blended logits
  are within ``NEAR_TIE`` (``near_tie``), where float rounding may flip the class.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES_DIR = GOLDEN_DIR / "cases"

# top-2 logit gap below which a class may legitimately differ between implementations
NEAR_TIE = 1e-3

TINY_MODELS = ("tiny0", "tiny1")
OCM_MODELS = "ocm_v4"  # OCM's released v4 ensemble (2 smp U-Nets), needs the weights
DEFAULT_OCM_WEIGHTS = Path(__file__).parents[1] / ".ocm_weights"  # git-ignored


class TinyNet(nn.Module):
    """Small fully convolutional net: (B, 3, H, W) -> (B, 4, H, W) logits, any H and W.

    Dilated convolutions give it a receptive field of 15 px, so seams between patches change
    its output, which is what the tiling tests need.
    """

    def __init__(self, in_channels: int = 3, classes: int = 4, width: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def init_tiny(seed: int) -> TinyNet:
    """A TinyNet with seeded weights, scaled so its logits span a few units."""
    torch.manual_seed(seed)
    model = TinyNet()
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(2.0)
    return model.eval()


def load_tiny(name: str) -> TinyNet:
    """Load a tiny model's weights from ``models.npz``."""
    with np.load(GOLDEN_DIR / "models.npz") as weights:
        state = {
            key.split("/", 1)[1]: torch.from_numpy(weights[key])
            for key in weights.files
            if key.startswith(name + "/")
        }
    if not state:
        raise KeyError(f"No weights for {name!r} in models.npz")
    model = TinyNet()
    model.load_state_dict(state)
    return model.eval()


def ocm_weights_dir() -> Path:
    """Where OCM's v4 weights are: ``$MANYPRED_OCM_WEIGHTS``, or ``.ocm_weights`` in the repo."""
    return Path(os.environ.get("MANYPRED_OCM_WEIGHTS", DEFAULT_OCM_WEIGHTS))


def load_ocm_models(weights_dir: str | Path | None = None) -> list[nn.Module]:
    """OCM's v4 ensemble, built as OCM builds it (smp U-Net, 3 in, 4 classes). Skips the
    calling test if the weights or segmentation-models-pytorch are missing."""
    import pytest

    smp = pytest.importorskip("segmentation_models_pytorch")
    from safetensors.torch import load_file

    weights_dir = Path(weights_dir) if weights_dir is not None else ocm_weights_dir()
    if not weights_dir.is_dir():
        pytest.skip(f"OCM weights not found in {weights_dir}, set MANYPRED_OCM_WEIGHTS")
    models = []
    for encoder, stem in (
        ("tu-regnety_004", "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state"),
        ("tu-edgenext_small", "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state"),
    ):
        model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=4)
        model.load_state_dict(load_file(weights_dir / f"{stem}.safetensors"))
        models.append(model.eval())
    return models


def load_inputs() -> dict[str, Any]:
    """Base crops and their georeferencing (``geo``: {name: {crs, transform}})."""
    with np.load(GOLDEN_DIR / "inputs.npz") as data:
        out: dict[str, Any] = {name: data[name] for name in data.files if name != "geo"}
        out["geo"] = json.loads(str(data["geo"]))
    return out


def convert_dtype(array: np.ndarray, kind: str) -> np.ndarray:
    """Turn a uint16 crop into the input dtype a case tests. Zeros (nodata) stay nodata."""
    if kind == "uint16":
        return array
    if kind == "uint8":
        return np.clip(array // 40, 0, 255).astype(np.uint8)
    if kind == "float32":
        return (array / 10000).astype(np.float32)
    if kind == "float32_nan":
        out = (array / 10000).astype(np.float32)
        out[:, (array == 0).all(axis=0)] = np.nan
        return out
    raise ValueError(f"unknown dtype kind {kind!r}")


def make_image(inputs: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    """One (C, H, W) image from {source, rows, cols, dtype}."""
    (r0, r1), (c0, c1) = spec["rows"], spec["cols"]
    crop = np.ascontiguousarray(inputs[spec["source"]][:, r0:r1, c0:c1])
    return convert_dtype(crop, spec["dtype"])


def make_chips(inputs: dict[str, Any], spec: dict[str, Any]) -> np.ndarray:
    """(N, C, S, S) chips from {chips: [[source, row, col], ...], size, dtype}."""
    size = spec["size"]
    chips = [inputs[src][:, r : r + size, c : c + size] for src, r, c in spec["chips"]]
    return convert_dtype(np.stack(chips), spec["dtype"])


def case_names() -> list[str]:
    return sorted(path.stem for path in CASES_DIR.glob("*.npz"))


def load_case(name: str) -> dict[str, Any]:
    """A case's meta (dict) and arrays (output, near_tie, mask, ...)."""
    with np.load(CASES_DIR / f"{name}.npz") as data:
        case: dict[str, Any] = {key: data[key] for key in data.files if key != "meta"}
        case["meta"] = json.loads(str(data["meta"]))
    return case
