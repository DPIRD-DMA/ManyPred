"""Blending patch predictions into a mosaic with OmniCloudMask's gradient weights.

Two ways to merge overlapping patches (``merge``, per output):

* ``"blend"`` (the default): the gradient-weighted mean of every patch covering a pixel.
* ``"cut"``: each pixel takes the prediction of the one patch whose gradient weight is largest
  there (its most central patch), all channels together; on equal weights the patch added
  first wins. The mosaic keeps the values and the largest weight so far, so it takes the same
  memory as blending. The weights are compared in the mosaic's dtype: in a float16 mosaic,
  weights within float16's precision of each other count as equal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .windows import Window

MERGES = ("blend", "cut")


def check_merge(merge: Any) -> None:
    """Raise unless ``merge`` is one of :data:`MERGES` or a dict of them."""
    values = merge.values() if isinstance(merge, Mapping) else [merge]
    for value in values:
        if not isinstance(value, str) or value not in MERGES:
            raise ValueError(f"merge must be one of {MERGES} (or a dict of them), got {value!r}.")


def gradient_weights(
    patch_size: int,
    overlap: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(patch_size, patch_size) blend weights: 1 in the middle, ramping linearly down to
    1/r over the outer r pixels on every side, where r is ``overlap`` capped at
    ``patch_size // 2`` (the product of a row and a column ramp, so 1/r**2 at the corners),
    so overlapping patches fade into each other. All ones without overlap. An overlap above
    half the patch ramps only over half of it: the weights still reach 1 in the middle, and
    edge pixels weigh 1/(patch_size // 2), not 1/overlap. Identical to OmniCloudMask's
    ``create_gradient_mask``."""
    if overlap <= 0:
        return torch.ones((patch_size, patch_size), dtype=dtype, device=device)
    overlap = min(overlap, patch_size // 2)
    ramp = torch.full((patch_size, patch_size), overlap, dtype=torch.int, device=device)
    ramp[:, :overlap] = torch.arange(1, overlap + 1, device=device).tile(patch_size, 1)
    ramp[:, -overlap:] = torch.arange(overlap, 0, -1, device=device).tile(patch_size, 1)
    ramp = ramp / overlap
    return (torch.rot90(ramp) * ramp).to(dtype)


def _paste(
    merge: str,
    sums: torch.Tensor,
    weight: torch.Tensor | None,
    prediction: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Merge one patch's (C, h, w) prediction (already weighted when blending) into the
    mosaic's (C, h, w) ``sums`` and (h, w) ``weight`` views, in place."""
    if merge == "blend":
        sums += prediction
        if weight is not None:
            weight += weights
        return
    # cut: replace where this patch is strictly more central, so ties keep the earlier patch
    better = weights > weight
    torch.where(better, prediction, sums, out=sums)  # in place: no (C, h, w) temporary
    torch.maximum(weight, weights, out=weight)


def _in_dtype(weights: torch.Tensor, device: Any, dtype: torch.dtype) -> torch.Tensor:
    """Patch weights in a mosaic's dtype. In float16 they stop at its smallest normal value:
    the weights near a patch's corners (down to 1/overlap**2) would lose precision, or round
    to 0, and a scene's corner pixels have nothing else to divide by."""
    if dtype == torch.float16:
        weights = weights.clamp(min=torch.finfo(dtype).tiny)
    return weights.to(device=device, dtype=dtype)


def _merged(merge: str, sums: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The float32 values of a mosaic's ``sums`` and ``weight``: divided when blending (so
    pixels no patch covered are 0 / 0 = NaN), and NaN where no patch covered when cutting."""
    if merge == "blend":
        return (sums / weight).float()
    return torch.where(weight > 0, sums.float(), torch.nan)


class MosaicBase:
    """What :class:`Mosaic` and :class:`BandMosaic` share: the merge rule, the patch weights
    in the mosaic's dtype (with a copy per prediction device) and the summed values."""

    merge: str
    sum: torch.Tensor
    weight: torch.Tensor | None

    def __init__(self, weights: torch.Tensor, device: torch.device, dtype: torch.dtype, merge: str):
        check_merge(merge)
        self.merge = merge
        self.weights = _in_dtype(weights, device, dtype)
        self._weight_copies: dict[torch.device, torch.Tensor] = {}

    def add(self, predictions: torch.Tensor, windows: list[Window]) -> None:
        raise NotImplementedError

    def merged(self, sums: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Float32 values of ``sums`` and ``weight`` from this mosaic (or rows of them)."""
        return _merged(self.merge, sums, weight)

    def _weights_on(self, device: torch.device) -> torch.Tensor:
        """The patch weights on ``device``, copied once per mosaic rather than every batch (the
        mosaic may live on the CPU while predictions come from the GPU)."""
        if self.weights.device == device:
            return self.weights
        if device not in self._weight_copies:
            self._weight_copies[device] = self.weights.to(device)
        return self._weight_copies[device]

    def _prepared(self, predictions: torch.Tensor) -> torch.Tensor:
        """Predictions in the mosaic's dtype, weighted when blending, on the mosaic's device."""
        prepared = predictions.to(self.sum.dtype)
        if self.merge == "blend":
            prepared = prepared * self._weights_on(predictions.device)
        return prepared.to(self.sum.device)


class Mosaic(MosaicBase):
    """Weighted sum of patch predictions and of their weights for one output, in ``dtype``
    (float32 by default); with ``merge="cut"``, the values of the most central patch so far and
    its weight instead. ``track_weight=False`` (blending only) keeps no summed weights, for
    outputs that only need the order of the sums (argmax), so ``blended`` is unavailable.

    ``add`` takes predictions (B, C, h, w) and their windows; ``blended`` divides the sum by
    the weights (as float32), so pixels no patch covered are NaN (0 / 0)."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        merge: str = "blend",
        track_weight: bool = True,
    ):
        super().__init__(weights, device, dtype, merge)
        if merge != "blend" and not track_weight:
            raise ValueError(f"merge={merge!r} compares weights, so it must track them.")
        self.sum = torch.zeros((channels, height, width), dtype=dtype, device=device)
        self.weight = (
            torch.zeros((height, width), dtype=dtype, device=device) if track_weight else None
        )

    def add(self, predictions: torch.Tensor, windows: list[Window]) -> None:
        """Add (B, C, h, w) predictions at their windows (on this mosaic's grid); parts of a
        window past the mosaic's edge are dropped."""
        prepared = self._prepared(predictions)
        height, width = self.sum.shape[-2:]
        # one window at a time, in order, so overlaps are summed (or ties broken) in a fixed
        # order
        for prediction, (top, bottom, left, right) in zip(prepared, windows, strict=True):
            rows, cols = min(bottom, height) - top, min(right, width) - left
            _paste(
                self.merge,
                self.sum[:, top : top + rows, left : left + cols],
                None if self.weight is None else self.weight[top : top + rows, left : left + cols],
                prediction[:, :rows, :cols],
                self.weights[:rows, :cols],
            )

    def blended(self) -> torch.Tensor:
        """(C, H, W) float32 merged values; NaN where no patch was added."""
        assert self.weight is not None, "a mosaic without weights has no merged values"
        return self.merged(self.sum, self.weight)

    def take_blended(self) -> torch.Tensor:
        """The values of :meth:`blended` in the mosaic's dtype (float32 with ``merge="cut"``),
        using up the mosaic: it divides in place and frees the weights."""
        if self.merge != "blend":
            return self.blended()
        assert self.weight is not None, "a mosaic without weights has no merged values"
        values = self.sum.div_(self.weight)
        self.sum, self.weight = None, None  # type: ignore[assignment]  # used up
        return values


class BandMosaic(MosaicBase):
    """A :class:`Mosaic` that holds only a band of rows: rows above ``top`` have been taken
    out with :meth:`take`, rows are added below as windows reach them. For scenes too large
    to blend in memory, with windows processed roughly top to bottom. It always tracks the
    summed weights."""

    weight: torch.Tensor

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        merge: str = "blend",
    ):
        super().__init__(weights, device, dtype, merge)
        self.height, self.width = height, width
        self.top = 0
        self.sum = torch.zeros((channels, 0, width), dtype=dtype, device=device)
        self.weight = torch.zeros((0, width), dtype=dtype, device=device)

    @property
    def bottom(self) -> int:
        return self.top + self.weight.shape[0]

    def _extend(self, bottom: int) -> None:
        bottom = min(bottom, self.height)
        if bottom > self.bottom:
            rows = bottom - self.bottom
            self.sum = torch.cat(
                [self.sum, self.sum.new_zeros((self.sum.shape[0], rows, self.width))], dim=1
            )
            self.weight = torch.cat([self.weight, self.weight.new_zeros((rows, self.width))])

    def add(self, predictions: torch.Tensor, windows: list[Window]) -> None:
        """As :meth:`Mosaic.add`; windows must not start above rows already taken."""
        prepared = self._prepared(predictions)
        for prediction, (top, bottom, left, right) in zip(prepared, windows, strict=True):
            if top < self.top:
                raise AssertionError(f"window at row {top} after rows < {self.top} were taken")
            self._extend(bottom)
            rows, cols = min(bottom, self.height) - top, min(right, self.width) - left
            local = top - self.top
            _paste(
                self.merge,
                self.sum[:, local : local + rows, left : left + cols],
                self.weight[local : local + rows, left : left + cols],
                prediction[:, :rows, :cols],
                self.weights[:rows, :cols],
            )

    def take(self, until: int) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        """Remove and return ``(top, sum, weight)`` for rows [top, until): rows no remaining
        window can reach. None if there is nothing new."""
        until = min(until, self.height)
        if until <= self.top:
            return None
        self._extend(until)
        rows = until - self.top
        taken = (self.top, self.sum[:, :rows].clone(), self.weight[:rows].clone())
        self.sum, self.weight = self.sum[:, rows:].clone(), self.weight[rows:].clone()
        self.top = until
        return taken
