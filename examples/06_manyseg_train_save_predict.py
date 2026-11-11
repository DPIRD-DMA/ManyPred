"""Train a ManySeg model, save it with what inference needs, and predict with ManyPred.

manyseg.save stores the normalisation the model was trained with, each head's activation and
the class names next to the weights. manyseg.load rebuilds the model, and ManyPred reads all
of that from the model's tiling_spec(), so predict_array needs no normalize= or activation=.
Derived outputs (here, counting objects) go in postprocess, after blending.

Needs ManySeg with save/load. Runs in seconds on the CPU (a tiny model, a few steps).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

try:
    import manyseg
    from manyseg.instances import connected_components, relabel_sequential
except ImportError:
    sys.exit("This example needs ManySeg (pip install manyseg).")
if not hasattr(manyseg, "save"):
    sys.exit("This example needs a ManySeg with manyseg.save / manyseg.load.")

from manypred import normalize_torch, predict_array

workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
torch.manual_seed(0)


def synthetic_scene(size):
    """A uint16 scene with bright square 'objects' (class 1) on a darker background."""
    image = rng.integers(800, 1500, size=(3, size, size)).astype(np.uint16)
    labels = np.zeros((size, size), dtype=np.int64)
    for _ in range(size // 16):
        r, c, s = rng.integers(0, size - 12), rng.integers(0, size - 12), rng.integers(5, 12)
        image[:, r : r + s, c : c + s] += 3000
        labels[r : r + s, c : c + s] = 1
    return image, labels


# 1. build and train (briefly) on 64 px chips, normalised as we will store it
NORMALIZE = [{"type": "scale", "value": 10000}, {"type": "zscore", "scope": "tile"}]
model = manyseg.create_model(
    backbone="resnet18",
    pretrained=False,
    in_channels=3,
    heads={"objects": {"type": "segmentation", "out_channels": 2}},
)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for _ in range(30):
    chips, targets = zip(*(synthetic_scene(64) for _ in range(8)), strict=True)
    x = normalize_torch(torch.from_numpy(np.stack(chips).astype(np.float32)), NORMALIZE)
    loss = torch.nn.functional.cross_entropy(
        model(x)["objects"], torch.from_numpy(np.stack(targets))
    )
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
print(f"training loss after 30 steps: {loss.item():.3f}")

# 2. save with what inference needs
path = manyseg.save(
    model,
    workdir / "objects.safetensors",
    normalize=NORMALIZE,
    activations={"objects": "softmax"},
    classes={"objects": ["background", "object"]},
    bands=["red", "green", "blue"],
)
print("saved", path.name, "| normalisation, activations and classes stored")


# 3. load and predict a larger scene, tiled; nothing about the model is repeated here
def count_objects(result):
    """Runs once on the blended result: label connected objects and count them."""
    mask = torch.from_numpy(result["objects"][0] == 1)[None]
    return int(relabel_sequential(connected_components(mask)).max())


loaded = manyseg.load(path)
scene, truth = synthetic_scene(300)
classes = predict_array(
    loaded, scene, patch_size=128, overlap=32, output="argmax", device="cpu", progress=False
)
count = predict_array(
    loaded,
    scene,
    patch_size=128,
    overlap=32,
    output="argmax",
    device="cpu",
    progress=False,
    postprocess=count_objects,
)
agreement = (classes["objects"][0] == truth).mean()
print(f"pixel agreement with the truth: {agreement:.3f}; objects counted: {count}")
assert classes["objects"].shape == (1, 300, 300)
