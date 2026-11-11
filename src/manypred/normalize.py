"""Input normalisation: nodata-aware, per band, computed in float32 (on the device in the
pipeline).

A normalisation spec is plain data, so it can be stored with a model and reused in training:
one step or a list of steps, applied in order, per input (a dict ``{input_name: steps}``)::

    [{"type": "scale", "value": 10000}, {"type": "zscore", "mean": [...], "std": [...]}]
    {"type": "zscore", "scope": "tile"}   # per patch (per chip in batch mode)

==============  =====================================================  =====================
type            options                                                result
==============  =====================================================  =====================
``none``                                                               ``x``
``scale``       ``value`` (scalar or per band)                         ``x / value``
``zscore``      ``mean`` and ``std`` (per band or scalar), or          ``(x - mean) / std``
                ``scope``: ``"tile"`` (per patch/chip) or ``"scene"``
                (whole scene, computed once before tiling)
``minmax``      ``min``, ``max`` (per band or scalar), ``clip=True``   to [0, 1]
``percentile``  ``low=2``, ``high=98``, ``scope`` (``"tile"``, the       stretch to [0, 1]
                default, or ``"scene"``), ``clip=True``
``db``          ``eps=1e-10`` (floor before the log)                   ``10 log10(x)``
``clip``        ``min``, ``max`` (scalar or per band, either           limits values
                optional)
==============  =====================================================  =====================

A step may also be a callable ``f(x, valid) -> x`` on a (B, C, H, W) float32 tensor and its
band-level valid mask. Statistics ignore nodata and NaN; a standard deviation (or percentile
range) of 0 counts as 1. Infinity counts as NaN. After the steps, nodata and NaN values are 0,
so a model never sees them, with no steps too (no spec, ``False`` or ``{"type": "none"}``).

:func:`normalize_torch` and :func:`normalize_numpy` apply a spec outside the pipeline (for
training code); there ``"tile"`` and ``"scene"`` both mean statistics of each image passed.

With a lazy reader, ``"scene"`` statistics come from a regular sample of at most 4 million pixels,
so they can differ slightly from an array's, which use every pixel.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch

Step = Mapping[str, Any] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
NormalizeSpec = Step | Sequence[Step] | None
Normalizer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_OPTIONS = {
    "none": set(),
    "scale": {"value"},
    "zscore": {"mean", "std", "scope"},
    "minmax": {"min", "max", "clip"},
    "percentile": {"low", "high", "scope", "clip"},
    "db": {"eps"},
    "clip": {"min", "max"},
}
_SCOPES = ("tile", "scene")


# --------------------------------------------------------------------------- validity


def band_valid_mask(x: torch.Tensor, nodata: float | None, may_be_nan: bool = True) -> torch.Tensor:
    """Bool mask, same shape as ``x``, of values that are not ``nodata``. NaN and infinity are
    never valid, whatever ``nodata`` is. ``may_be_nan=False`` (data that came from an integer
    dtype) skips those checks. Integer ``x`` is compared exactly (a float ``nodata`` would
    make torch compare in float32, where 16777216 equals 16777217)."""
    if not x.is_floating_point() and not x.is_complex():
        return _integer_valid(x, nodata)
    if not may_be_nan:
        if nodata is None or (isinstance(nodata, float) and math.isnan(nodata)):
            return torch.ones_like(x, dtype=torch.bool)
        return x != nodata
    if nodata is None:
        return torch.isfinite(x) if x.is_floating_point() else torch.ones_like(x, dtype=torch.bool)
    if isinstance(nodata, float) and math.isnan(nodata):
        return torch.isfinite(x)
    valid = x != nodata
    if x.is_floating_point():
        valid &= torch.isfinite(x)
    return valid


def _integer_valid(x: torch.Tensor, nodata: float | None) -> torch.Tensor:
    """:func:`band_valid_mask` of integer (or bool) data: only a whole ``nodata`` within the
    dtype's range can match."""
    if nodata is None or isinstance(nodata, bool):
        return torch.ones_like(x, dtype=torch.bool)
    if isinstance(nodata, (int, np.integer)):
        value = int(nodata)  # kept exact: a float rounds sentinels past 2**53
    else:
        number = float(nodata)
        if not math.isfinite(number) or not number.is_integer():
            return torch.ones_like(x, dtype=torch.bool)
        value = int(number)
    if x.dtype == torch.bool:
        return x != bool(value) if value in (0, 1) else torch.ones_like(x, dtype=torch.bool)
    info = torch.iinfo(x.dtype)
    if not info.min <= value <= info.max:
        return torch.ones_like(x, dtype=torch.bool)
    if x.dtype in (torch.uint16, torch.uint32):
        x = x.to(torch.int64)  # older torch has no comparisons for these
    return x != value


# --------------------------------------------------------------------------- validation


def steps_of(spec: NormalizeSpec) -> list[Step]:
    """The steps of a spec (None or False -> no steps)."""
    if spec is None or spec is False:
        return []
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        return list(spec)
    return [spec]


def validate_spec(spec: NormalizeSpec) -> list[Step]:
    """Check a spec and return its steps; raises ValueError saying what to fix."""
    steps = steps_of(spec)
    for step in steps:
        if callable(step):
            continue
        if not isinstance(step, Mapping):
            raise ValueError(
                f"A normalisation step must be a dict like {{'type': 'scale', 'value': 255}} "
                f"or a callable f(x, valid) -> x, got {step!r}."
            )
        if "type" not in step:
            raise ValueError(f"Normalisation step {dict(step)!r} has no 'type' key.")
        kind = step["type"]
        if kind not in _OPTIONS:
            raise ValueError(
                f"unknown normalisation type {kind!r}; use one of {sorted(_OPTIONS)} or a callable."
            )
        extra = set(step) - _OPTIONS[kind] - {"type"}
        if extra:
            raise ValueError(
                f"{kind!r} step has unexpected keys {sorted(extra)}; allowed: "
                f"{sorted(_OPTIONS[kind])}."
            )
        scope = step.get("scope")
        if scope is not None and scope not in _SCOPES:
            raise ValueError(f"{kind!r} step: scope must be one of {_SCOPES}, got {scope!r}.")
        if kind == "scale" and "value" not in step:
            raise ValueError("'scale' step needs 'value' (scalar or one per band).")
        if kind == "zscore":
            has_stats = "mean" in step and "std" in step
            if has_stats == (scope is not None) or (("mean" in step) != ("std" in step)):
                raise ValueError(
                    "'zscore' step needs dataset mean and std, or scope='tile' / 'scene' "
                    "(not both)."
                )
        if kind == "minmax":
            for key in ("min", "max"):
                if key not in step:
                    raise ValueError(f"'minmax' step needs '{key}'.")
        if kind == "scale":
            _check_values(step["value"], "scale value", nonzero=True)
        if kind == "zscore" and has_stats:
            _check_values(step["mean"], "mean")
            _check_values(step["std"], "std", positive=True)
        if kind == "minmax":
            lo, hi = _check_values(step["min"], "min"), _check_values(step["max"], "max")
            if lo.size != hi.size and 1 not in (lo.size, hi.size):
                raise ValueError("'minmax' min and max need the same number of values.")
            if not np.all(lo < hi):
                raise ValueError("'minmax' step needs min < max for every band.")
        if kind == "clip":
            if "min" not in step and "max" not in step:
                raise ValueError("'clip' step needs 'min', 'max' or both.")
            lo = _check_values(step["min"], "clip min") if "min" in step else None
            hi = _check_values(step["max"], "clip max") if "max" in step else None
            if (
                lo is not None
                and hi is not None
                and lo.size != hi.size
                and 1
                not in (
                    lo.size,
                    hi.size,
                )
            ):
                raise ValueError("'clip' min and max need the same number of values.")
            if lo is not None and hi is not None and not np.all(lo <= hi):
                raise ValueError("'clip' step needs min <= max.")
        if kind == "db" and not (_is_number(step.get("eps", 1e-10)) and step.get("eps", 1e-10) > 0):
            raise ValueError(f"'db' step needs eps > 0, got {step.get('eps')}.")
        if "clip" in step and not isinstance(step["clip"], bool):
            raise ValueError(f"{kind!r} step: clip must be True or False, got {step['clip']!r}.")
        if kind == "percentile":
            low, high = step.get("low", 2), step.get("high", 98)
            if not all(_is_number(v) for v in (low, high)) or not 0 <= low < high <= 100:
                raise ValueError(
                    f"'percentile' step needs 0 <= low < high <= 100, got low={low}, high={high}."
                )
    seen_per_tile = False
    for step in steps:
        scope = step.get("scope") if isinstance(step, Mapping) else None
        if scope == "scene" and seen_per_tile:
            raise ValueError(
                "A scope='scene' step can't follow a per-tile step (scope='tile', a percentile "
                "step without scope, or a callable): scene statistics are computed once over "
                "the whole scene, before tiling. Put scene steps first."
            )
        seen_per_tile |= _per_tile(step)
    return steps


def _per_tile(step: Step) -> bool:
    """Whether a step's result depends on the patch it sees: statistics per tile (a
    ``percentile`` without ``scope`` is per tile), or a callable, which may be."""
    if callable(step):
        return True
    scope = step.get("scope")
    return scope == "tile" or (step.get("type") == "percentile" and scope is None)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _check_values(
    value: Any, name: str, *, nonzero: bool = False, positive: bool = False
) -> np.ndarray:
    """A scalar or per-band list as an array, checked to be finite (and nonzero/positive)."""
    try:
        array = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or a list of numbers, got {value!r}.") from None
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numbers, got {value!r}.")
    if nonzero and np.any(array == 0):
        raise ValueError(f"{name} must not be 0, got {value!r}.")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return array


def has_scene_scope(spec: NormalizeSpec) -> bool:
    return any(
        isinstance(step, Mapping) and step.get("scope") == "scene" for step in steps_of(spec)
    )


# --------------------------------------------------------------------------- statistics


def _per_band(value: Any, x: torch.Tensor, name: str) -> torch.Tensor:
    """A scalar or per-band value as a (1, C, 1, 1) float32 tensor on x's device."""
    t = torch.as_tensor(value, dtype=torch.float32, device=x.device).flatten()
    if t.numel() == 1:
        return t.view(1, 1, 1, 1)
    if t.numel() != x.shape[1]:
        raise ValueError(
            f"{name} has {t.numel()} values but the input has {x.shape[1]} bands; give one "
            f"value per band or a scalar."
        )
    return t.view(1, -1, 1, 1)


def zscore_tile(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per image and band: subtract the mean and divide by the standard deviation of the valid
    values; invalid values become 0. ``x`` is (B, C, H, W) float32. Population std, with a std
    of 0 treated as 1."""
    count = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mean = torch.where(valid, x, 0).sum(dim=(2, 3), keepdim=True) / count
    centered = torch.where(valid, x - mean, 0)
    std = ((centered**2).sum(dim=(2, 3), keepdim=True) / count).sqrt()
    std = torch.where(std == 0, 1, std)
    return centered / std


def percentiles(x: torch.Tensor, valid: torch.Tensor, qs: Sequence[float]) -> list[torch.Tensor]:
    """Per image and band percentiles of the valid values, (B, C, 1, 1) each, with numpy's
    default linear interpolation. Images/bands without valid values give 0."""
    b, c = x.shape[:2]
    flat = torch.where(valid, x, torch.inf).reshape(b * c, -1)
    ordered, _ = torch.sort(flat, dim=1)
    count = valid.reshape(b * c, -1).sum(dim=1)
    # float64 so positions stay exact past 2**24 values; MPS has no float64
    position_dtype = torch.float32 if x.device.type == "mps" else torch.float64
    last = (count - 1).clamp(min=0).to(position_dtype)
    out = []
    for q in qs:
        position = last * (q / 100)
        lo = position.floor().long()
        hi = position.ceil().long()
        frac = (position - lo).to(torch.float32)
        low_value = ordered.gather(1, lo[:, None])[:, 0]
        high_value = ordered.gather(1, hi[:, None])[:, 0]
        value = low_value + (high_value - low_value) * frac
        value = torch.where(count > 0, value, 0)
        out.append(value.view(b, c, 1, 1))
    return out


def _stretch(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, clip: bool) -> torch.Tensor:
    span = hi - lo
    out = (x - lo) / torch.where(span == 0, 1, span)
    return out.clamp(0, 1) if clip else out


def _apply_step(step: Step, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if callable(step):
        return step(x, valid)
    kind = step["type"]
    if kind == "none":
        return x
    if kind == "scale":
        return x / _per_band(step["value"], x, "scale value")
    if kind == "zscore":
        if "mean" in step:
            return (x - _per_band(step["mean"], x, "mean")) / _per_band(step["std"], x, "std")
        return zscore_tile(x, valid)
    if kind == "minmax":
        lo, hi = _per_band(step["min"], x, "min"), _per_band(step["max"], x, "max")
        return _stretch(x, lo, hi, step.get("clip", True))
    if kind == "percentile":
        lo, hi = percentiles(x, valid, (step.get("low", 2), step.get("high", 98)))
        return _stretch(x, lo, hi, step.get("clip", True))
    if kind == "db":
        return 10 * torch.log10(x.clamp(min=step.get("eps", 1e-10)))
    if kind == "clip":
        if "min" in step:
            x = torch.maximum(x, _per_band(step["min"], x, "clip min"))
        if "max" in step:
            x = torch.minimum(x, _per_band(step["max"], x, "clip max"))
        return x
    raise AssertionError(kind)


def _check_batch_mode(steps: list[Step], batch_mode: bool) -> None:
    if batch_mode and any(isinstance(s, Mapping) and s.get("scope") == "scene" for s in steps):
        raise ValueError(
            "scope='scene' needs a whole scene; in batch mode each chip is normalised on its "
            "own, so use scope='tile' or dataset statistics."
        )


def build_normalizer(spec: NormalizeSpec, *, batch_mode: bool = False) -> Normalizer:
    """``f(x, valid) -> x`` applying ``spec``'s steps in order to a (B, C, H, W) float32
    batch, then setting invalid values to 0 (all it does for ``None``). ``"scene"`` steps must
    have been resolved (:func:`resolve_scene`) before."""
    steps = validate_spec(spec)
    _check_batch_mode(steps, batch_mode)

    def normalize(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        for step in steps:
            x = _apply_step(step, x, valid)
            # values a step made non-finite (a callable, or float32 overflow such as
            # 1 / 1e-40) are invalid for later statistics and become 0
            valid = valid & torch.isfinite(x)
        return torch.where(valid, x, 0)

    return normalize


# values per row chunk when scene statistics stream over an image
SCENE_CHUNK = 1 << 24


def resolve_scene(spec: NormalizeSpec, image: Any, nodata: float | None) -> Any:
    """Replace ``scope="scene"`` steps by equivalent steps with the statistics of ``image``
    ((C, H, W), or (T, C, H, W) for a time series, pooling the frames; a tensor or array of
    any dtype), computed after the steps before them: ``zscore`` gets ``mean``/``std``, and
    ``percentile`` becomes ``minmax``. Other specs are returned unchanged.

    The image is read in row chunks, so memory is a chunk's float32 copy (and, for
    ``percentile``, one band's valid values) rather than copies of the whole scene."""
    if not has_scene_scope(spec):
        return spec
    steps = validate_spec(spec)
    resolved: list[Step] = []
    for step in steps:
        if isinstance(step, Mapping) and step.get("scope") == "scene":
            if step["type"] == "zscore":
                step = _scene_zscore(image, nodata, resolved)
            else:
                step = _scene_minmax(image, nodata, resolved, step)
        resolved.append(step)
    return resolved


def _scene_chunks(
    image: Any, nodata: float | None, before: list[Step]
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """``(x, valid)`` for row chunks of ``image`` as (N, C, h, W) float32 after the steps
    ``before`` (all per value: scene steps come before tile steps and callables)."""
    height = image.shape[-2]
    per_row = max(1, math.prod(image.shape[:-2]) * image.shape[-1])
    rows = max(1, SCENE_CHUNK // per_row)
    for top in range(0, height, rows):
        chunk = image[..., top : top + rows, :]
        if not isinstance(chunk, torch.Tensor):
            chunk = np.asarray(chunk)
            chunk = torch.from_numpy(
                np.ascontiguousarray(chunk, dtype=chunk.dtype.newbyteorder("="))
            )
        chunk = chunk if chunk.ndim == 4 else chunk[None]
        valid = band_valid_mask(chunk, nodata)  # in the original dtype, before rounding
        x = chunk.to(torch.float32)
        for step in before:
            x = _apply_step(step, x, valid)
            valid = valid & torch.isfinite(x)
        yield x, valid


def _band_sums(x: torch.Tensor) -> torch.Tensor:
    """Per band sums of a (N, C, h, W) chunk, as float64 on the CPU (MPS has no float64, so a
    chunk is summed there in float32)."""
    if x.device.type == "mps":
        return x.sum(dim=(0, 2, 3)).cpu().double()
    return x.sum(dim=(0, 2, 3), dtype=torch.float64).cpu()


def _scene_zscore(image: Any, nodata: float | None, before: list[Step]) -> Step:
    """A zscore step with the scene's per band mean and population std (two passes, summed
    in float64 on the CPU)."""
    count = total = None
    for x, valid in _scene_chunks(image, nodata, before):
        part = _band_sums(torch.where(valid, x, 0))
        n = valid.sum(dim=(0, 2, 3)).cpu()
        total, count = (part, n) if total is None else (total + part, count + n)
    assert total is not None and count is not None
    count = count.clamp(min=1).to(torch.float64)
    mean = (total / count).to(torch.float32)
    squares = None
    for x, valid in _scene_chunks(image, nodata, before):
        centered = torch.where(valid, x - mean.to(x.device).view(1, -1, 1, 1), 0)
        part = _band_sums(centered**2)
        squares = part if squares is None else squares + part
    assert squares is not None
    std = (squares / count).sqrt().to(torch.float32)
    std = torch.where(std == 0, 1, std)
    return {"type": "zscore", "mean": mean.tolist(), "std": std.tolist()}


def _scene_minmax(image: Any, nodata: float | None, before: list[Step], step: Step) -> Step:
    """A percentile step as a minmax step with the scene's per band percentiles, pooling
    frames; band by band, so memory is one band's valid values."""
    qs = (step.get("low", 2), step.get("high", 98))
    channels = image.shape[-3]
    lo, hi = torch.zeros(channels), torch.zeros(channels)
    for band in range(channels):
        parts = [x[:, band][valid[:, band]] for x, valid in _scene_chunks(image, nodata, before)]
        values = torch.cat(parts)
        del parts
        if values.numel():
            lo[band], hi[band] = _percentiles_of(values, qs)
    # float64, so lo + 1 differs from lo for large values (a constant band)
    lo, hi = lo.double(), hi.double()
    hi = torch.where(
        hi == lo, torch.maximum(lo + 1, torch.nextafter(lo, torch.full_like(lo, math.inf))), hi
    )
    return {
        "type": "minmax",
        "min": lo.tolist(),
        "max": hi.tolist(),
        "clip": step.get("clip", True),
    }


def _percentiles_of(values: torch.Tensor, qs: Sequence[float]) -> list[torch.Tensor]:
    """Percentiles of a 1D tensor with numpy's linear interpolation, with ``kthvalue`` (no
    sort of the whole tensor)."""
    out = []
    for q in qs:
        position = (values.numel() - 1) * q / 100
        lo, frac = int(math.floor(position)), position - math.floor(position)
        low = values.kthvalue(lo + 1).values
        high = values.kthvalue(min(lo + 2, values.numel())).values if frac else low
        out.append(low + (high - low) * frac)
    return out


# --------------------------------------------------------------------------- public helpers


def normalize_torch(
    x: torch.Tensor, spec: NormalizeSpec, nodata: float | None = None
) -> torch.Tensor:
    """Apply a spec to a (C, H, W) or (B, C, H, W) tensor (any dtype, any device): float32
    out, nodata and NaN as 0. ``"tile"`` and ``"scene"`` statistics are per image."""
    batched = x.ndim == 4
    if x.ndim not in (3, 4):
        raise ValueError(f"x must be (C, H, W) or (B, C, H, W), got shape {tuple(x.shape)}.")
    x = x if batched else x[None]
    valid = band_valid_mask(x, nodata)  # in the original dtype, before rounding to float32
    x = x.to(torch.float32)
    steps = [
        {**s, "scope": "tile"} if isinstance(s, Mapping) and s.get("scope") == "scene" else s
        for s in validate_spec(spec)
    ]
    out = build_normalizer(steps)(x, valid)
    return out if batched else out[0]


def normalize_numpy(x: np.ndarray, spec: NormalizeSpec, nodata: float | None = None) -> np.ndarray:
    """:func:`normalize_torch` for numpy arrays (same implementation), float32 out."""
    x = np.asarray(x)
    if x.dtype in (np.uint16, np.uint32):
        x = x.astype(np.int64)  # exact, and older torch supports few uint16/uint32 operations
    return normalize_torch(torch.from_numpy(np.ascontiguousarray(x)), spec, nodata).numpy()
