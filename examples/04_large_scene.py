"""A scene too large for memory: read lazily, written in row blocks.

load_func returns a RasterioReader instead of an array, so patches are read as needed, and
the default GeoTIFF writer receives finished rows as soon as no remaining patch can change
them. Peak memory depends on the scene width and patch size, not its height.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

from manypred import predict_files
from manypred.geo import RasterioReader

workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
path = workdir / "tall_scene.tif"
height, width = 4000, 600
profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 3,
    "dtype": "uint16",
    "crs": "EPSG:32750",
    "tiled": True,
    "transform": Affine(10, 0, 400000, 0, -10, 6500000),
}
rng = np.random.default_rng(0)
with rasterio.open(path, "w", **profile) as dst:
    for top in range(0, height, 500):  # written in strips, never whole
        strip = rng.integers(1, 10000, size=(3, 500, width), dtype=np.uint16)
        dst.write(strip, window=rasterio.windows.Window(0, top, width, 500))


def load_lazy(item):
    reader = RasterioReader(item)
    return reader, reader.profile


torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(16, 4, 1)
)
[output] = predict_files(
    model,
    [path],
    load_func=load_lazy,
    output_dir=workdir / "predictions",
    patch_size=256,
    overlap=64,
    batch_size=4,
    device="cpu",
    output="argmax",
    normalize={"type": "zscore", "scope": "tile"},
)
with rasterio.open(output) as src:
    print(output.name, src.shape, src.dtypes[0])
    assert src.shape == (height, width)
