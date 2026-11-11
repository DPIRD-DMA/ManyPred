"""OmniCloudMask's options expressed with ManyPred (for porting OCM pipelines).

OCM normalises each patch with a nodata-aware z-score, blends raw logits and then, for
confidence maps, applies a softmax, adds 0.001 and clips to [0.001, 0.999]. ManyPred
reproduces this with a normalisation spec and a callable output (applied to the blended logits
before masking).
Here a small untrained model stands in for OCM's ensemble.
"""

import numpy as np
import torch

from manypred import predict_array

OCM_NORMALIZE = {"type": "zscore", "scope": "tile"}


def ocm_confidence(logits: torch.Tensor) -> torch.Tensor:
    """export_confidence=True, softmax_output=True."""
    # in place after the softmax: on a whole scene, each new array would take gigabytes
    probs = torch.softmax(logits, dim=1).add_(0.001).clip_(0.001, 0.999)
    return probs.nan_to_num_(nan=0.0)


torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(16, 4, 1)
)
scene = np.random.default_rng(0).integers(1, 10000, size=(3, 600, 700), dtype=np.uint16)
kwargs = {
    "patch_size": 256,
    "overlap": 64,
    "normalize": OCM_NORMALIZE,
    "nodata": 0,
    "dtype": "float32",  # OCM's default inference_dtype, and ManyPred's
    "adapt_patch_size": True,  # OCM shrinks patches over large nodata areas
    "device": "cpu",
    "progress": False,
}

mask = predict_array(model, scene, output="argmax", **kwargs)  # export_confidence=False
logits = predict_array(model, scene, output="logits", **kwargs)  # softmax_output=False
confidence = predict_array(model, scene, output=ocm_confidence, **kwargs)  # softmax_output=True
print(mask.shape, mask.dtype, logits.shape, confidence.min().round(3), confidence.max().round(3))
