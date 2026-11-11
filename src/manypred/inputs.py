"""Model inputs: kinds, resolution ratios to the primary input, and their patch windows.

Inputs come as one (C, H, W) array or a dict. In a dict:

* a (C, h, w) array is a **raster** at ``scale = h / H`` of the primary input (e.g. 0.5 for a
  20 m band beside a 10 m primary); its window is the primary window times the scale;
* a (T, C, h, w) array is a **time series**, tiled in h and w only;
* a (D,) array is a **vector** (metadata, or a time series' ``<name>_mask`` / ``<name>_dates``),
  passed through untiled and batched, unnormalised, in its own dtype (floats as the
  inference dtype).

The primary input is the first raster or time series (as in ManySeg). Scales are inferred
from the shapes or given with ``scales={name: scale}`` (then each side may be the floor or ceil
of the scaled primary side; with the floor, the last primary rows or columns have no data from
that input and see padding). Ratios must be whole numbers or 1 / whole numbers: a patch at
another ratio can't be resampled exactly as the whole scene is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .collate import SINGLE
from .windows import Window


@dataclass(frozen=True)
class InputInfo:
    name: str
    kind: str  # "raster", "timeseries" or "vector"
    scale: Fraction = Fraction(1)

    @property
    def tiled(self) -> bool:
        return self.kind != "vector"


# How close (relatively) a ratio must be to the nearest whole number k or 1 / k to be taken as
# it: rounded decimals such as 0.333 or 0.0167 (1/60) are; 2/3, 0.3 or 1.5 are far from any
_RATIO_TOLERANCE = 0.02


def ratio(value: Any) -> Fraction:
    """A resolution ratio as an exact fraction: the whole number k or 1 / k nearest to
    ``value`` when it is within 2% of it (``1 / 3``, ``0.333`` and ``"1/3"`` give 1/3, ``0.01``
    gives 1/100). Any other positive value gives a fraction that is neither, which callers
    refuse (``2 / 3`` gives 2/3). Raises TypeError, ValueError or OverflowError for a value
    that isn't a finite number."""
    if isinstance(value, str):
        try:
            value = Fraction(value)
        except ZeroDivisionError:
            raise ValueError(f"{value!r} isn't a number.") from None
    value = float(value)
    if value > 0 and math.isfinite(value):
        try:
            if value >= 1:
                nearest = Fraction(round(value))
            else:  # the closer of the two 1 / k around it
                k = 1 / value
                nearest = min(
                    (Fraction(1, n) for n in {max(1, math.floor(k)), math.ceil(k)}),
                    key=lambda f: abs(value - f),
                )
        except (OverflowError, ZeroDivisionError):  # too small to be a ratio of two grids
            nearest = None
        if nearest is not None and abs(value - nearest) <= _RATIO_TOLERANCE * nearest:
            return nearest
    simple = Fraction(value).limit_denominator(1000)  # 2/3 rather than its float's fraction
    return simple if math.isclose(simple, value, rel_tol=1e-6) else Fraction(value)


def _kind(name: str, ndim: int) -> str:
    kinds = {1: "vector", 3: "raster", 4: "timeseries"}
    if ndim not in kinds:
        raise ValueError(
            f"Input {name!r} has {ndim} dimensions; inputs must be (C, H, W) rasters, "
            f"(T, C, H, W) time series or (D,) vectors."
        )
    return kinds[ndim]


def describe_inputs(
    inputs: Mapping[str, Any],
    scales: Mapping[str, float] | None = None,
    order: Sequence[str] = (),
) -> tuple[str, dict[str, InputInfo]]:
    """The primary input's name and every input's kind and scale. The primary is the first
    raster or time series in ``order`` (the model's spec), else in ``inputs``.

    A single unnamed input (an array, not a dict) is the primary, so its scale is 1: one scale
    under the model's name for it (from its ``tiling_spec()``) is ignored."""
    scales = dict(scales or {})
    if set(inputs) == {SINGLE} and len(scales) == 1:
        scales = {}
    unknown = sorted(set(scales) - set(inputs))
    if unknown:
        raise ValueError(f"scales given for unknown inputs {unknown}; inputs: {list(inputs)}.")
    kinds = {name: _kind(name, len(x.shape)) for name, x in inputs.items()}
    rasters = [name for name, kind in kinds.items() if kind in ("raster", "timeseries")]
    for name in rasters:
        if 0 in tuple(inputs[name].shape):
            label = "The image" if name == SINGLE else f"Input {name!r}"
            raise ValueError(
                f"{label} is empty, shape {tuple(inputs[name].shape)}; it needs at least one "
                f"band and one pixel."
            )
    if not rasters:
        raise ValueError("At least one input must be a (C, H, W) raster or (T, C, H, W) series.")
    primary = next((name for name in order if name in rasters), rasters[0])
    height, width = inputs[primary].shape[-2:]
    infos = {}
    for name, x in inputs.items():
        kind = kinds[name]
        if kind == "vector":
            infos[name] = InputInfo(name, kind)
            continue
        h, w = x.shape[-2:]
        if name in scales:
            scale = ratio(scales[name])
            ok = all(
                size in (math.floor(base * scale), math.ceil(base * scale))
                for size, base in ((h, height), (w, width))
            )
        else:
            scale = Fraction(h, height)
            ok = Fraction(w, width) == scale
            if not ok or 1 not in (scale.numerator, scale.denominator):
                # sizes rounded per side (e.g. 52 x 52 or 6 x 8 for a half-resolution input
                # beside 103 x 103 or 13 x 17): the ratio 1/k or k that matches both sides
                # within floor/ceil, if exactly one does
                fits = [
                    candidate
                    for k in range(1, 17)
                    for candidate in {Fraction(1, k), Fraction(k)}
                    if all(
                        size in (math.floor(base * candidate), math.ceil(base * candidate))
                        for size, base in ((h, height), (w, width))
                    )
                ]
                if len(fits) == 1:
                    scale, ok = fits[0], True
        if ok and scale > 0 and 1 not in (scale.numerator, scale.denominator):
            raise ValueError(
                f"Input {name!r} is at {scale} of the primary input's resolution; ratios must "
                f"be whole numbers or 1 / whole numbers (patches at other ratios can't be "
                f"resampled like the whole scene)."
            )
        if not ok or scale <= 0:
            label = "The input" if name == SINGLE else f"Input {name!r}"
            if name in scales:
                raise ValueError(
                    f"{label} is {h} x {w}, which is not scale {scales[name]} of the "
                    f"{height} x {width} primary input {primary!r}. The scale comes from scales= "
                    f"or the model's tiling_spec(), where a raster input's scale defaults to 1; "
                    f"pass the right ratio, or scale None in the spec to take it from the shapes."
                )
            raise ValueError(
                f"{label} is {h} x {w} for a {height} x {width} primary input "
                f"{primary!r}: not one resolution ratio for both sides. Pass "
                f"scales={{{name!r}: ratio}} if it covers the same extent with rounded sizes."
            )
        infos[name] = InputInfo(name, kind, scale)
    return primary, infos


def check_channels_first(shape: tuple[int, ...], what: str) -> None:
    """Refuse a (H, W, C) image passed where (C, H, W) is expected: a last axis of at most 4
    after a first axis over 16 reads as a scene 4 or fewer pixels wide, which is almost
    certainly channels-last data."""
    if len(shape) == 3 and shape[-1] <= 4 and shape[0] > 16:
        raise ValueError(
            f"{what} must be (C, H, W), found shape {tuple(shape)}, which looks like (H, W, C). "
            f"Pass np.moveaxis(image, -1, 0) (or tensor.permute(2, 0, 1))."
        )


def window_multiple(infos: Mapping[str, InputInfo]) -> int:
    """Primary windows must start at multiples of this so every input's window is whole."""
    return math.lcm(*(info.scale.denominator for info in infos.values() if info.tiled))


def scale_window(window: Window, scale: Fraction) -> Window:
    top, bottom, left, right = window
    return tuple(int(v * scale) for v in (top, bottom, left, right))  # type: ignore[return-value]
