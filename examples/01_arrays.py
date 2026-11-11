"""Arrays in, arrays out: one scene tiled and blended, and a batch of chips.

Runs in seconds on the CPU with a small untrained model; swap in your own.
"""

import numpy as np
import torch

from manypred import predict_array, predict_batch

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(16, 4, 1)
)

# a (C, H, W) uint16 scene with a nodata corner (0 in every band)
rng = np.random.default_rng(0)
scene = rng.integers(1, 10000, size=(3, 700, 900), dtype=np.uint16)
scene[:, :150, :200] = 0

common = {
    "device": "cpu",
    "normalize": {"type": "zscore", "scope": "tile"},  # per-patch z-score, like OmniCloudMask
    "nodata": 0,
}

# tiled: 256 px patches overlapping by 64 px, blended with gradient weights
classes = predict_array(model, scene, patch_size=256, overlap=64, output="argmax", **common)
probs = predict_array(
    model, scene, patch_size=256, overlap=64, activation="softmax", output="probs", **common
)
print("classes", classes.shape, classes.dtype, "probs", probs.shape, probs.dtype)
assert classes.shape == (1, 700, 900) and probs.shape == (4, 700, 900)
assert np.isnan(probs[:, :150, :200]).all()  # nodata is masked
np.testing.assert_allclose(probs[:, 150:].sum(axis=0), 1, rtol=1e-5)

# batch: chips predicted whole, 8 at a time
chips = rng.integers(1, 10000, size=(20, 3, 128, 128), dtype=np.uint16)
logits = predict_batch(model, chips, batch_size=8, **common)
print("batch logits", logits.shape)
assert logits.shape == (20, 4, 128, 128)

# chips of different sizes: bucketed by shape, results in input order
mixed = [chips[0], chips[1, :, :96, :64], chips[2]]
out = predict_batch(model, mixed, group_by_shape=True, output="argmax", **common)
print("mixed sizes", [o.shape for o in out])
