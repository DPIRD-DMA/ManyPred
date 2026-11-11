"""Lazy scenes: readers that load windows on demand, and the masks and statistics built from
them without holding the scene in memory.

A reader is any object with ``shape`` (C, H, W), or (T, C, H, W) for a time series, and
``read(window) -> (C, h, w)`` (or (T, C, h, w)) array for
``window = (top, bottom, left, right)`` inside the scene (e.g.
:class:`manypred.geo.RasterioReader`). ``load_func`` may return one instead of an array.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .io_lock import IO_LOCK
from .windows import Window


@runtime_checkable
class Reader(Protocol):
    shape: tuple[int, ...]

    def read(self, window: Window) -> Any: ...


def is_reader(x: Any) -> bool:
    """A lazy reader rather than an array or tensor. Raises TypeError for an object with
    ``read`` and ``shape`` whose shape isn't (C, H, W) or (T, C, H, W), such as an open
    rasterio dataset."""
    if isinstance(x, (np.ndarray, torch.Tensor)) or not isinstance(x, Reader):
        return False
    shape = tuple(x.shape)
    if len(shape) not in (3, 4):
        hint = (
            " An open rasterio dataset isn't one: return manypred.geo.RasterioReader(path) "
            "from load_func instead."
            if type(x).__module__.startswith("rasterio")
            else ""
        )
        raise TypeError(
            f"{type(x).__name__} has read() and shape like a lazy reader, but its shape "
            f"{shape} isn't (C, H, W) or (T, C, H, W).{hint}"
        )
    return True


class ArrayReader:
    """A reader over an in-memory (C, H, W) or (T, C, H, W) array: the smallest reader, for
    trying a pipeline built for lazy scenes on an array. ``reads`` counts the calls to
    :meth:`read`."""

    def __init__(self, array: Any):
        self.array = array
        self.shape = tuple(array.shape)
        self.dtype = array.dtype
        self.reads = 0

    def read(self, window: Window) -> Any:
        self.reads += 1
        top, bottom, left, right = window
        return self.array[..., top:bottom, left:right]


def reader_profile(image: Any) -> Any:
    """The ``profile`` of a scene's primary input, on whose grid outputs are written, if it is a
    mapping describing that grid, as a :class:`~manypred.geo.RasterioReader`'s is."""
    profile = getattr(image, "profile", None)
    if not isinstance(profile, Mapping):
        return None
    height, width = image.shape[-2:]
    grid = profile.get("height", height), profile.get("width", width)
    return profile if grid == (height, width) else None


def read_readers(
    inputs: dict[str, Any], meta: Any, order: Sequence[str] = ()
) -> tuple[dict[str, Any], Any]:
    """Batch mode predicts chips whole, so a loaded chip's readers are read whole. ``meta``
    defaults to the primary reader's profile, as in tiled mode; the primary is the first
    raster or time series in ``order`` (the model's spec), else in ``inputs``."""
    readers = {name for name, x in inputs.items() if is_reader(x)}
    if not readers:
        return inputs, meta
    if meta is None:
        rasters = [name for name, x in inputs.items() if len(x.shape) in (3, 4)]
        primary = next((name for name in order if name in rasters), rasters[0])
        meta = reader_profile(inputs[primary]) if primary in readers else None
    whole = {
        name: read_window(x, (0, x.shape[-2], 0, x.shape[-1])) if name in readers else x
        for name, x in inputs.items()
    }
    return whole, meta


def read_window(reader: Any, window: Window) -> np.ndarray:
    """A reader's pixels under a window; reads take turns with saves (see :mod:`.io_lock`)."""
    with IO_LOCK.read():
        return np.asarray(reader.read(window))


def row_bands(height: int, rows: int) -> list[tuple[int, int]]:
    return [(top, min(top + rows, height)) for top in range(0, height, rows)]


def band_rows(reader: Any, budget: int = 64 * 2**20) -> int:
    """Rows per band so one band of the scene (every frame of a time series) is about
    ``budget`` bytes."""
    *planes, _, width = reader.shape
    itemsize = np.dtype(getattr(reader, "dtype", np.float32)).itemsize
    return max(1, budget // max(1, math.prod(planes) * width * itemsize))


class PackedMask:
    """An (H, W) bool mask stored as bits (1 bit per pixel); slicing returns bool arrays."""

    def __init__(self, packed: np.ndarray, width: int):
        self.packed = packed
        self.shape = (packed.shape[0], width)

    def __getitem__(self, key: Any) -> np.ndarray:
        rows, cols = key if isinstance(key, tuple) else (key, slice(None))
        block = self.packed[rows]
        if block.ndim == 1:
            return np.unpackbits(block, count=self.shape[1]).astype(bool)[cols]
        return np.unpackbits(block, axis=1, count=self.shape[1]).astype(bool)[:, cols]

    def any(self) -> bool:
        return bool(self.packed.any())

    def unpacked(self) -> np.ndarray:
        return self[:, :]


def nodata_mask(reader: Any, nodata: float | None) -> PackedMask:
    """Bit-packed (H, W) map of pixels where every band is nodata (or NaN), read in row
    bands."""
    from .windows import nodata_pixels

    _, height, width = reader.shape[-3:]
    packed = np.zeros((height, math.ceil(width / 8)), dtype=np.uint8)
    for top, bottom in row_bands(height, band_rows(reader)):
        block = read_window(reader, (top, bottom, 0, width))
        block = block.reshape(-1, *block.shape[-2:])  # a time series: every frame and band
        packed[top:bottom] = np.packbits(nodata_pixels(block, nodata), axis=1)
    return PackedMask(packed, width)


def nodata_fraction(reader: Any, nodata: float | None) -> float:
    """Fraction of values that are missing (:func:`manypred.windows.nodata_fraction`), read in
    row bands. OmniCloudMask's rule for shrinking patches."""
    from .windows import nodata_fraction as fraction

    _, height, width = reader.shape[-3:]
    total = 0.0
    for top, bottom in row_bands(height, band_rows(reader)):
        total += fraction(read_window(reader, (top, bottom, 0, width)), nodata) * (bottom - top)
    return total / height


def decimated(reader: Any, max_pixels: int = 4_000_000) -> np.ndarray:
    """The scene sampled every k-th row and column so it has at most ``max_pixels`` pixels
    (for scene-scope statistics), read in row bands."""
    _, height, width = reader.shape[-3:]
    step = max(1, math.ceil(math.sqrt(height * width / max_pixels)))
    while -(-height // step) * -(-width // step) > max_pixels:  # very narrow scenes
        step += 1
    rows = []
    for top, bottom in row_bands(height, max(step, band_rows(reader) // step * step)):
        block = read_window(reader, (top, bottom, 0, width))
        first = (-top) % step
        rows.append(block[..., first::step, ::step])
    return np.concatenate(rows, axis=-2)
