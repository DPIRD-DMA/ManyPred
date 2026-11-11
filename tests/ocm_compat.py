"""ManyPred configured like OmniCloudMask, for the parity tests.

OmniCloudMask (OCM) option mapping:

* ``export_confidence=False`` -> ``output="argmax"``;
* ``export_confidence=True, softmax_output=False`` -> ``output="logits"``;
* ``export_confidence=True, softmax_output=True`` -> ``output=ocm_confidence``: OCM applies the
  softmax to the blended logits and then clips ``p + 0.001`` to [0.001, 0.999], which a
  callable output (applied after blending, before nodata masking) reproduces. ManyPred's own
  ``output="probs"`` averages probabilities instead (activation before blending).

OCM normalises each patch (or chip) with its nodata-aware z-score: ``zscore`` ``scope="tile"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

OCM_NORMALIZE = {"type": "zscore", "scope": "tile"}


def ocm_confidence(x: torch.Tensor) -> torch.Tensor:
    """OCM's ``postprocess_preds`` with ``softmax_output=True`` on (B, C, H, W) logits."""
    probs = torch.clip(torch.softmax(x, dim=1) + 0.001, 0.001, 0.999)
    return torch.nan_to_num(probs, nan=0.0)


def missing_as_nan(ocm: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """A float output of OCM with its masked nodata pixels as ManyPred has them: OCM gives 0
    where a patch covered them (and NaN elsewhere), ManyPred always NaN."""
    return np.where(np.isnan(pred) & (ocm == 0), np.nan, ocm)


def manypred_kwargs(meta: dict[str, Any]) -> dict[str, Any]:
    """ManyPred keyword arguments equivalent to a golden case's OCM arguments."""
    kwargs = meta["kwargs"]
    out: dict[str, Any] = {
        "normalize": OCM_NORMALIZE,
        "nodata": kwargs.get("no_data_value", 0),
        "mask_nodata": kwargs.get("apply_no_data_mask", True),
        "batch_size": kwargs.get("batch_size", 1),
        # OCM's default, and ManyPred's
        "dtype": kwargs.get("inference_dtype", "float32"),
        "output": {"argmax": "argmax", "logits": "logits", "probs": ocm_confidence}[meta["mode"]],
        "progress": False,
    }
    if meta["func"] != "predict_from_batch":
        out["patch_size"] = kwargs["patch_size"]
        out["overlap"] = kwargs["patch_overlap"]
        out["adapt_patch_size"] = True  # OCM shrinks patches over large nodata areas
    return out
