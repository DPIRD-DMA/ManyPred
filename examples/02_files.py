"""Files in, files out: GeoTIFF scenes loaded, predicted and written as GeoTIFFs.

Needs the [geo] extra (rasterio). The next scene loads while the current one is predicted,
and outputs are written on a background thread.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

from manypred import predict_files

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(16, 4, 1)
)
workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())

# write three small georeferenced scenes
rng = np.random.default_rng(0)
paths = []
for i in range(3):
    path = workdir / f"scene_{i}.tif"
    data = rng.integers(1, 10000, size=(3, 300, 400), dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "height": 300,
        "width": 400,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32750",
        "transform": Affine(10, 0, 400000 + i * 4000, 0, -10, 6500000),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    paths.append(path)


def load(path):
    """load_func: returns the image and a meta object for save_func (here the profile)."""
    with rasterio.open(path) as src:
        return src.read(), src.profile


outputs = predict_files(
    model,
    paths,
    load_func=load,
    output_dir=workdir / "predictions",
    patch_size=128,
    overlap=32,
    device="cpu",
    output="argmax",
    normalize={"type": "zscore", "scope": "tile"},
)
for path in outputs:
    with rasterio.open(path) as src:
        print(path.name, src.shape, src.dtypes[0], src.crs)

# a second run with overwrite=False skips scenes that already have an output
again = predict_files(
    model,
    paths,
    load_func=load,
    output_dir=workdir / "predictions",
    overwrite=False,
    device="cpu",
    progress=False,
)
assert again == outputs
