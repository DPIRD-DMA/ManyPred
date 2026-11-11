"""Stitch with merge="cut", then post-process the saved rasters block by block.

Run 1 writes the model's outputs as GeoTIFFs. With merge="cut" every pixel comes from the most
central patch covering it, so nothing is averaged across patch edges. Run 2 reads the saved
raster window by window with postprocess_blocks and calls a function per block (core plus a
halo of context): here a morphological opening of the thresholded extent, which needs 2 pixels
of context, then the area left in the block's core. The blocks' areas add up to the opening of
the whole scene. A vector decoder works the same way (keeping the objects whose first pixel is
in the core), and can be re-tuned without running the model again.

Run 3 does both in one call: BlockPostprocess runs the same function on each block while the
scene is predicted, from a strip of finished rows rather than the saved file, and hands the
combined result to save (where a decoder would write a GeoPackage).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

from manypred import BlockPostprocess, postprocess_blocks, predict_files
from manypred.blocks import Block, Scene
from manypred.geo import RasterioReader


class Model(torch.nn.Module):
    """A stand-in segmentation model: an "extent" probability per pixel."""

    def forward(self, x):
        return {"extent": torch.sigmoid((x.float() - 550) / 50)}


def load(item):
    reader = RasterioReader(item)
    return reader, reader.profile


def opened(mask: np.ndarray) -> np.ndarray:
    """3 x 3 opening (erode, then dilate), with False beyond the array."""

    def filtered(x: np.ndarray, reduce) -> np.ndarray:
        padded = np.pad(x, 1)
        shifts = [
            padded[i : i + x.shape[0], j : j + x.shape[1]] for i in range(3) for j in range(3)
        ]
        return reduce(np.stack(shifts), axis=0)

    return filtered(filtered(mask, np.all), np.any)


def core_area(fields: dict[str, np.ndarray], block: Block, scene: Scene) -> int:
    """Pixels of the opened extent in this block's core. Module level, so the worker processes
    can import it."""
    mask = opened(fields["extent"][0] > 0.5)
    top, bottom, left, right = block.inner  # the core within the window
    return int(mask[top:bottom, left:right].sum())


def total_area(results: list[int], scene: Scene) -> float:
    """Square metres, from the raster's transform."""
    return sum(results) * abs(scene.transform.a * scene.transform.e)


def main(workdir: Path) -> None:
    path = workdir / "scene.tif"
    height, width = 700, 900
    rng = np.random.default_rng(0)
    image = np.full((1, height, width), 100, dtype=np.uint16)
    for _ in range(300):  # bright rectangles, some a pixel thin (the opening removes those)
        r, c = rng.integers(0, height - 30), rng.integers(0, width - 30)
        image[:, r : r + rng.integers(1, 30), c : c + rng.integers(1, 30)] = 1000
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:32750",
        "transform": Affine(0.5, 0, 400000, 0, -0.5, 6500000),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image)

    # run 1: stitch, cut rather than blended; written in row blocks (the reader is lazy)
    predict_files(
        Model(),
        [path],
        load_func=load,
        output_dir=workdir / "predictions",
        patch_size=256,
        overlap=64,
        device="cpu",
        merge="cut",
        progress=False,
    )
    extent = workdir / "predictions" / "scene_pred_extent.tif"

    # run 2: per block from the saved raster, in two worker processes
    area = postprocess_blocks(
        {"extent": extent}, core_area, block=256, halo=8, combine=total_area, workers=2
    )
    with rasterio.open(extent) as src:
        whole = opened(src.read(1) > 0.5).sum() * 0.25
    print(f"opened extent: {area:.2f} m2 from blocks, {whole:.2f} m2 from the whole scene")
    assert area == whole

    # run 3: the same while predicting; the GeoTIFF is still written as before
    saved = {}
    post = BlockPostprocess(
        core_area,
        block=256,
        halo=8,
        combine=total_area,
        workers=2,
        save=lambda result, item, scene: saved.setdefault(item, result),
    )
    predict_files(
        Model(),
        [path],
        load_func=load,
        output_dir=workdir / "predictions_hooked",
        patch_size=256,
        overlap=64,
        device="cpu",
        merge="cut",
        block_postprocess=post,
        progress=False,
    )
    print(f"opened extent while predicting: {saved[path]:.2f} m2")
    assert saved[path] == area


if __name__ == "__main__":  # worker processes import this file: run only as a script
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()))
