"""Patch windows over a scene: the grid, clamping at the edges, shifting away from nodata."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .devices import warn_caller
from .normalize import band_valid_mask

# (top, bottom, left, right) in scene pixels
Window = tuple[int, int, int, int]

# OmniCloudMask's smallest patch size when it shrinks patches over large nodata areas
MIN_SHRUNK_PATCH = 32


def make_patch_indexes(
    height: int,
    width: int,
    patch_size: int,
    overlap: int,
    multiple: int = 1,
    pad: bool = False,
) -> list[Window]:
    """Row-major grid of ``patch_size`` windows with stride ``patch_size - overlap``; the last
    row and column are clamped to the scene edge (so they overlap more, and the list can hold
    duplicates, which :func:`plan_windows` drops). With ``multiple`` > 1 (models with
    reduced-resolution outputs) every window starts at a multiple of it, so the last row and
    column are clamped to the first multiple at or past ``size - patch_size``: they may extend
    past the scene by less than ``multiple`` pixels (and are padded when read). With ``pad``,
    the patch may be larger than the scene (a model with a fixed input size): windows then
    start at 0 in that dimension and extend past the scene."""
    if not 0 <= overlap < patch_size:
        raise ValueError(
            f"overlap must be at least 0 and less than patch_size, got overlap={overlap}, "
            f"patch_size={patch_size}."
        )
    if not pad and patch_size > min(height, width):
        raise ValueError(f"patch_size {patch_size} is larger than the scene {(height, width)}.")
    stride = patch_size - overlap
    last_top = max(0, -(-(height - patch_size) // multiple) * multiple)
    last_left = max(0, -(-(width - patch_size) // multiple) * multiple)
    windows = []
    for top in range(0, height, stride):
        top = min(top, last_top)
        for left in range(0, width, stride):
            left = min(left, last_left)
            windows.append((top, top + patch_size, left, left + patch_size))
    return windows


def window_block(nodata: Any, window: Window) -> np.ndarray:
    """``nodata[top:bottom, left:right]`` as an (h, w) bool array, clipped to the map. A
    bit-packed map (:class:`~manypred.lazy.PackedMask`) unpacks only the window's columns."""
    top, bottom, left, right = window
    height, width = nodata.shape
    top, bottom = max(top, 0), min(bottom, height)
    left, right = max(left, 0), min(right, width)
    packed = getattr(nodata, "packed", None)
    if packed is None:
        return np.asarray(nodata[top:bottom, left:right])
    first = left // 8
    bits = np.unpackbits(packed[top:bottom, first : -(-right // 8)], axis=1)
    return bits[:, left - 8 * first : right - 8 * first].view(bool)


def nodata_pixels(image: np.ndarray | torch.Tensor, nodata: float | None) -> np.ndarray:
    """(H, W) bool: pixels where every band is ``nodata``, NaN or infinite."""
    if isinstance(image, torch.Tensor):
        # integers compared exactly: torch compares int32 with a float nodata in float32,
        # where 16777216 equals 16777217
        return (~band_valid_mask(image, nodata)).all(dim=0).numpy(force=True)
    image = np.asarray(image)
    floating = np.issubdtype(image.dtype, np.floating)
    if nodata is None and not floating:
        return np.zeros(image.shape[1:], dtype=bool)
    nan = nodata is None or (isinstance(nodata, float) and math.isnan(nodata))

    def missing(band: np.ndarray) -> np.ndarray:
        if nan:
            return ~np.isfinite(band)
        out = band == nodata
        if floating:
            out |= ~np.isfinite(band)
        return out

    # band by band in row chunks: no (C, H, W) temporary, and each chunk stays in cache
    height = image.shape[1]
    result = np.empty(image.shape[1:], dtype=bool)
    rows = max(1, (1 << 22) // max(1, image.shape[2]))
    for top in range(0, height, rows):
        chunk = image[:, top : top + rows]
        block = missing(chunk[0])
        for band in chunk[1:]:
            block &= missing(band)
        result[top : top + rows] = block
    return result


def shift_window(nodata: np.ndarray, window: Window) -> Window:
    """Move a window inward while an edge row or column is entirely nodata (OmniCloudMask's
    ``get_patch``): down while the top row is nodata, else up while the bottom row is; then
    right while the left column is, else left while the right column is. Windows stay inside
    the scene."""
    top, bottom, left, right = window
    max_bottom, max_right = nodata.shape
    if not nodata[top:bottom, left:right].any():
        return window

    moved = False
    while bottom < max_bottom and nodata[top, left:right].all():
        top, bottom, moved = top + 1, bottom + 1, True
    if not moved:
        while top > 0 and nodata[bottom - 1, left:right].all():
            top, bottom = top - 1, bottom - 1

    moved = False
    while right < max_right and nodata[top:bottom, left].all():
        left, right, moved = left + 1, right + 1, True
    if not moved:
        while left > 0 and nodata[top:bottom, right - 1].all():
            left, right = left - 1, right - 1
    return top, bottom, left, right


def _round_toward(value: int, start: int, multiple: int) -> int:
    """``value`` moved to a multiple of ``multiple``, rounding toward ``start`` (itself a
    multiple), so a shifted window never gives up rows or columns of where it started."""
    if value >= start:
        return value - value % multiple
    return -(-value // multiple) * multiple


def _shifted(nodata: Any, window: Window, multiple: int) -> Window:
    """:func:`shift_window` on the part of ``nodata`` it can reach, then back on multiples of
    ``multiple``. A window with valid data moves less than its size in each direction (it
    stops at the latest when that data reaches its edge), so a bit-packed map only unpacks
    the window and a patch's width around it."""
    top, bottom, left, right = window
    height, width = nodata.shape
    size = bottom - top
    row0, col0 = max(0, top - size + 1), max(0, left - size + 1)
    near = window_block(nodata, (row0, bottom + size - 1, col0, right + size - 1))
    shifted = shift_window(near, (top - row0, bottom - row0, left - col0, right - col0))
    new_top = _round_toward(shifted[0] + row0, top, multiple)
    new_left = _round_toward(shifted[2] + col0, left, multiple)
    return new_top, new_top + size, new_left, new_left + (right - left)


def plan_windows(
    nodata: Any,
    patch_size: int,
    overlap: int,
    shift: bool = True,
    multiple: int = 1,
    pad: bool = False,
) -> list[Window]:
    """The windows to predict, in order: the grid, minus windows that are entirely nodata,
    shifted away from nodata edges (then moved back to a multiple of ``multiple``, toward
    where they started; windows past the scene edge are not shifted), without duplicates.
    ``nodata`` is an (H, W) bool array or a bit-packed map; ``pad``: see
    :func:`make_patch_indexes`."""
    height, width = nodata.shape
    planned, seen = [], set()
    for window in make_patch_indexes(height, width, patch_size, overlap, multiple, pad):
        _, bottom, _, right = window
        block = window_block(nodata, window)
        missing = np.count_nonzero(block)
        if missing == block.size:
            continue
        if shift and missing and bottom <= height and right <= width:
            window = _shifted(nodata, window, multiple)
        if window not in seen:
            seen.add(window)
            planned.append(window)
    return planned


def adjust_patch_size(
    height: int,
    width: int,
    nodata_fraction: float,
    patch_size: int,
    overlap: int,
    multiple: int = 1,
    adapt: bool = True,
    fixed: bool = False,
) -> tuple[int, int]:
    """OmniCloudMask's ``check_patch_size``: shrink the patch (and overlap) when the scene is
    smaller than it, or, with ``adapt``, when over 30 % of the values are nodata and the patch
    is over half the scene (so fewer patches are mostly nodata). Warns when it changes them.
    ``fixed`` (a model with a fixed input size): the patch is never changed; windows larger
    than the scene are padded instead (see :func:`make_patch_indexes`)."""
    if overlap < 0 or overlap >= patch_size:
        raise ValueError(
            f"overlap must be at least 0 and less than patch_size, got overlap={overlap}, "
            f"patch_size={patch_size}."
        )
    if fixed:
        return patch_size, overlap
    short = min(height, width)
    if adapt and nodata_fraction > 0.3 and patch_size > short / 2:
        patch_size = max(short // 2, min(MIN_SHRUNK_PATCH, short))
        overlap = min(overlap, patch_size // 2)
        warn_caller(
            f"Over 30% of the scene is nodata; using patch_size={patch_size} and "
            f"overlap={overlap} so fewer patches are mostly nodata."
        )
    if patch_size > short:
        patch_size = short
        overlap = min(overlap, patch_size // 2)
        warn_caller(
            f"patch_size is larger than the scene {(height, width)}; using "
            f"patch_size={patch_size} and overlap={overlap}."
        )
    if multiple > 1:
        # shrunk sizes stay multiples of the coarsest output scale
        if patch_size < multiple:
            raise ValueError(
                f"The scene {(height, width)} is smaller than one {multiple} x {multiple} "
                f"block of the model's coarsest output."
            )
        patch_size -= patch_size % multiple
        overlap -= overlap % multiple
        overlap = min(overlap, patch_size - multiple)
    return patch_size, overlap


def nodata_fraction(image: np.ndarray | torch.Tensor, nodata: float | None) -> float:
    """Fraction of values (all bands) that are missing: equal to ``nodata``, NaN or infinite.
    OmniCloudMask's rule for ``adapt_patch_size``, also counting NaN as missing."""
    value = nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata))
    if isinstance(image, torch.Tensor):
        missing = ~band_valid_mask(image, nodata)  # integers compared exactly
        return int(torch.count_nonzero(missing)) / image.numel()
    image = np.asarray(image)
    floating = np.issubdtype(image.dtype, np.floating)
    count = 0
    # in row chunks: no temporaries the size of the scene
    rows = max(1, (1 << 22) // max(1, image.shape[-1]))
    for top in range(0, image.shape[-2], rows):
        chunk = image[..., top : top + rows, :]
        missing = ~np.isfinite(chunk) if floating else np.zeros(chunk.shape, dtype=bool)
        if value:
            missing |= chunk == nodata
        count += int(np.count_nonzero(missing))
    return count / image.size
