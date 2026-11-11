"""Model outputs: activations, output modes and nodata masking.

Each model output goes through, in order:

1. ``activation`` (``"softmax"``, ``"sigmoid"``, a callable on the (B, C, ...) tensor,
   ``"none"`` or None), per patch or chip, before any blending, so probabilities are averaged,
   not logits. None uses the activation the model declares, if any; ``"none"`` means no activation;
2. blending (tiled mode) or nothing (batch mode); pixels nothing was predicted for (all-nodata
   patches and chips) are NaN;
3. ``output``: ``"auto"`` (the default: ``"probs"`` when the output has an activation, given
   or declared by the model's ``tiling_spec()``, else ``"logits"``), ``"logits"`` (the values
   as they are, no activation allowed), ``"probs"`` (needs an activation; an output the model
   declares as ``"none"`` keeps its values), ``"argmax"`` (uint8 class index, int64 past 256
   classes, ``(1, H, W)``), or
   a callable applied to the blended float32 array (``(1, C, H, W)`` in tiled mode, or
   ``(B, C, H, W)`` in batch mode). An ``"argmax"`` given for every output leaves outputs the
   model declares as regression as they are, with a warning;
4. nodata masking: nodata pixels are NaN in float outputs and 0 in class outputs.

``"argmax"`` outputs have no value reserved for nodata: nodata pixels (and chips or patches
without any data) are class 0, the same value as a real prediction of the first class. Keep
the input's nodata mask (or ask for ``"probs"``/``"logits"``, where nodata is NaN) to tell
them apart. ``"argmax"`` of NaN at a pixel with data is an error: the model failed, and class 0
would hide it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

Activation = str | Callable[[torch.Tensor], torch.Tensor] | None
OutputMode = str | Callable[[torch.Tensor], torch.Tensor]

_ACTIVATIONS = ("softmax", "sigmoid")
_MODES = ("logits", "probs", "argmax")


def pairwise_argmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Argmax along ``dim`` (kept as size 1) by pairwise comparison, lower index on ties.
    Faster than ``torch.argmax`` on the CPU and MPS for few classes."""
    dim = dim % x.ndim
    best = x.select(dim, 0)
    index = torch.zeros_like(best, dtype=torch.int64)
    for i in range(1, x.size(dim)):
        candidate = x.select(dim, i)
        better = ~(best >= candidate)
        index = torch.where(better, i, index)
        best = torch.where(better, candidate, best)
    return index.unsqueeze(dim)


# Values per piece when argmax goes in pieces, so its int64 temporaries stay small
FINISH_CHUNK = 1 << 22


def _has_nan(x: torch.Tensor) -> bool:
    """Whether a float tensor holds NaN. aminmax passes NaN on and makes no temporaries, so a
    whole mosaic costs no memory."""
    return x.numel() > 0 and bool(torch.isnan(torch.stack(torch.aminmax(x))).any())


def argmax(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Class index along ``dim`` (kept), as uint8 when there are at most 256 classes, else
    int64.

    NaN (no prediction) counts as 0, so a pixel without a prediction is class 0, which looks
    the same as a real prediction of class 0. Uses
    :func:`pairwise_argmax` on the CPU and MPS and ``torch.argmax`` elsewhere. Large arrays go
    in pieces, so the int64 indexes these return never take more than a little memory."""
    dim = dim % x.ndim
    if x.numel() > 0 and not torch.isfinite(torch.stack(torch.aminmax(x))).all():
        x = torch.nan_to_num(x, nan=0.0)
    find = (
        pairwise_argmax
        if x.device.type in ("cpu", "mps")
        else lambda t, dim: torch.argmax(t, dim=dim, keepdim=True)
    )
    if x.size(dim) > 256:
        return find(x, dim=dim)
    others = [d for d in range(x.ndim) if d != dim]
    across = max(reversed(others), key=x.size) if others else dim
    step = max(1, FINISH_CHUNK * x.size(across) // max(1, x.numel()))
    # MPS: one piece, as its memory is shared with the host and slicing it has been unreliable
    if x.device.type == "mps" or step >= x.size(across):
        return find(x, dim=dim).to(torch.uint8)
    shape = list(x.shape)
    shape[dim] = 1
    out = torch.empty(shape, dtype=torch.uint8, device=x.device)
    for start in range(0, x.size(across), step):
        piece = x.narrow(across, start, min(step, x.size(across) - start))
        out.narrow(across, start, piece.size(across)).copy_(find(piece, dim=dim))
    return out


def _label(name: str) -> str:
    return f"Output {name!r}" if name else "The model output"


@dataclass(frozen=True)
class OutputSpec:
    """How one model output is treated."""

    name: str
    activation: Activation = None
    mode: OutputMode = "logits"
    # whether raw values were asked for (``output`` given, or ``activation="none"``), rather
    # than the "logits" default; image-level outputs need it (see check_shape)
    explicit: bool = True
    named: bool = False  # whether ``mode`` was given for this output by name (a dict)
    not_classes: bool = False  # a global "argmax" left this regression output's values alone

    def __post_init__(self) -> None:
        act, mode = self.activation, self.mode
        if act is not None and not callable(act) and act not in _ACTIVATIONS:
            raise ValueError(
                f"Output {self.name!r}: activation must be one of {_ACTIVATIONS}, a callable "
                f"or None, got {act!r}."
            )
        if not callable(mode) and mode not in _MODES:
            raise ValueError(
                f"Output {self.name!r}: output must be one of {('auto', *_MODES)} or a callable, "
                f"got {mode!r}."
            )
        if mode == "logits" and act is not None:
            raise ValueError(
                f"Output {self.name!r}: output='logits' returns the raw model output, but "
                f"activation={act!r} is set. Use output='probs' for activated values."
            )
        if mode == "probs" and act is None:
            raise ValueError(
                f"Output {self.name!r}: output='probs' needs an activation; pass "
                f"activation='softmax' (one class per pixel) or 'sigmoid' (multi-label)."
            )

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation along the class dimension (1) of a (B, C, ...) float32
        tensor."""
        if self.activation is None:
            return x
        if self.activation == "softmax":
            return torch.softmax(x, dim=1)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        return self.activation(x)

    def check_shape(self, ndim: int) -> None:
        """Image-level (B, C) outputs may be multi-class or multi-label, so their treatment
        must be chosen explicitly."""
        if ndim == 2 and self.activation is None and not self.explicit:
            raise ValueError(
                f"{_label(self.name)} is image-level (B, C): pass activation='softmax' "
                f"(multi-class) or 'sigmoid' (multi-label) with output='probs', or "
                f"output='logits' for raw values."
            )

    @property
    def is_class(self) -> bool:
        return self.mode == "argmax"

    def finalize(self, x: torch.Tensor, class_dim: int) -> torch.Tensor:
        """Turn a blended (or batch) array (float32, or the mosaic's dtype for argmax) into the
        requested output."""
        if self.mode == "argmax":
            return argmax(x, dim=class_dim)
        if callable(self.mode):
            return self.mode(x)
        return x


def per_output(value: Any, names: list[str], what: str, alias: str | None = None) -> dict[str, Any]:
    """Expand a setting that may be given once or as ``{output_name: value}``."""
    if isinstance(value, Mapping):
        if names == [""] and len(value) == 1:
            # a single-output model returns a bare tensor; settings may use its head's name
            key = next(iter(value))
            if alias is not None and key not in ("", alias):
                raise ValueError(f"{what} given for {key!r}, but the model's output is {alias!r}.")
            if alias is None and key != "":
                raise ValueError(
                    f"{what} given for {key!r}, but the model returns a single unnamed "
                    f"tensor; pass {what} as a plain value."
                )
            return {"": value[key]}
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f"{what} given for unknown outputs {unknown}; outputs: {names}.")
        return {name: value.get(name) for name in names}
    return {name: value for name in names}


def make_specs(
    names: list[str],
    activation: Any,
    output: Any,
    defaults: Mapping[str, str] | None = None,
    spec_names: list[str] | None = None,
    regression: Sequence[str] = (),
) -> dict[str, OutputSpec]:
    """One :class:`OutputSpec` per output name.

    ``defaults``: ``{output: "softmax" | "sigmoid" | "none"}`` from the model (its
    ``tiling_spec()``), used for outputs given no activation. With ``output="auto"`` (or
    None) an output with an activation (given or declared) gives ``"probs"``, else
    ``"logits"``.
    ``output="logits"`` or ``activation="none"`` asks for raw values explicitly, whatever the
    model says. ``regression``: outputs the model declares as regression, where an ``"argmax"``
    not given by name would make meaningless classes, so they keep their values."""
    alias = spec_names[0] if spec_names and len(spec_names) == 1 else None
    activations = per_output(activation, names, "activation", alias)
    modes = per_output(output, names, "output", alias)
    defaults = dict(defaults or {})
    if names == [""] and len(defaults) == 1:
        # a single-output model returns a bare tensor; its spec names the head
        defaults = {"": next(iter(defaults.values()))}
    regression = set(regression)
    if names == [""] and spec_names and len(spec_names) == 1 and spec_names[0] in regression:
        regression = {""}
    specs = {}
    for name in names:
        act, mode = activations[name], modes[name]
        if mode == "auto":
            mode = None  # the model's declared activation decides, below
        explicit = mode is not None
        if act is None and name in defaults and mode != "logits":
            # an explicit output="logits" asks for raw values, whatever the model says
            act = defaults[name]
            if act == "none" and mode == "probs":
                # the model says this output is already the prediction (a regression head)
                mode = "logits"
        if act == "none":
            act, explicit = None, True
        if act is not None and mode is None:
            # "auto" with an activation (given or declared) gives the activated values
            mode = "probs"
        named = isinstance(output, Mapping) and (
            name in output or (name == "" and len(output) == 1)
        )
        not_classes = mode == "argmax" and not named and name in regression
        if not_classes:
            mode = "probs" if act is not None else "logits"
        specs[name] = OutputSpec(name, act, mode or "logits", explicit, named, not_classes)
    return specs


def mask_nodata(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mark nodata pixels: NaN in float outputs (as pixels without a prediction, so a missing
    pixel doesn't depend on where the patches fell), 0 in class outputs. ``valid`` (bool)
    broadcasts against ``x``."""
    if x.is_floating_point():
        return torch.where(valid, x, torch.nan)
    return torch.where(valid, x, 0)


def check_argmax(x: torch.Tensor, data: torch.Tensor, name: str) -> None:
    """Raise if (B, C, ...) values about to go through ``"argmax"`` hold NaN where the input
    has data (``data``, bool, broadcasting against them): argmax would make it class 0, a
    normal-looking answer for a model that failed."""
    if not _has_nan(x):
        return
    count = int(torch.count_nonzero(torch.isnan(x).any(dim=1, keepdim=True) & data.to(x.device)))
    if count:
        raise RuntimeError(
            f"{_label(name)}: the model returned NaN at {count} pixels with data, so "
            f"output='argmax' has no answer there. Check the model with output='logits', and "
            f"try dtype='float32' if it runs in a 16-bit dtype."
        )


@dataclass
class TilePredictions:
    """Image-level outputs of a tiled prediction: one vector per patch.

    Attributes:
        values: (N, C) float32, per patch after activation.
        windows: (N, 4) top, bottom, left, right in scene pixels (clipped to the scene).
        valid: (N,) fraction of the patch's pixels that are valid.
        shape: (H, W) of the scene.
        weights: (P, P) blend weights of a patch.
        valid_map: (H, W) bool valid pixels of the scene (an array, or a bit-packed
            :class:`~manypred.lazy.PackedMask` for lazy scenes), or None if all are valid.
    """

    values: np.ndarray
    windows: np.ndarray
    valid: np.ndarray
    shape: tuple[int, int]
    weights: np.ndarray
    valid_map: Any = None

    def to_map(self, stride: int | None = None, rows: tuple[int, int] | None = None) -> np.ndarray:
        """(C, H, W) map (or (C, ceil(H/stride), ceil(W/stride)), sampling every stride-th
        pixel): each patch's vector painted over its window with the blend weights, divided by
        the weight sum. NaN where no patch was predicted and at nodata pixels (``valid_map``).
        ``rows=(start, stop)`` gives only those rows of the (strided) map."""
        stride = stride or 1
        height, width = (-(-self.shape[0] // stride), -(-self.shape[1] // stride))
        start, stop = rows or (0, height)
        total = np.zeros((self.values.shape[1], stop - start, width), dtype=np.float64)
        weight = np.zeros((stop - start, width), dtype=np.float64)
        for value, (top, bottom, left, right) in zip(self.values, self.windows, strict=True):
            rows_hit = np.arange(max(-(-top // stride), start), min(-(-bottom // stride), stop))
            if rows_hit.size == 0:
                continue
            cols = np.arange(-(-left // stride), -(-right // stride))
            w = self.weights[np.ix_(rows_hit * stride - top, cols * stride - left)]
            local = rows_hit - start
            total[:, local[:, None], cols[None, :]] += value[:, None, None] * w
            weight[local[:, None], cols[None, :]] += w
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (total / weight).astype(np.float32)
        if self.valid_map is not None:
            valid = np.asarray(
                self.valid_map[start * stride : stop * stride : stride, ::stride], dtype=bool
            )
            out[:, ~valid] = np.nan
        return out

    def reduce(self, how: str = "mean", min_valid: float = 0.0) -> np.ndarray:
        """(C,) summary of the scene.

        ``"mean"``: mean of :meth:`to_map` over valid pixels (so overlaps and nodata are
        handled), skipping NaN pixels of the map (a patch's NaN value makes every pixel it
        covers NaN, overlapping patches included). ``"max"``: max over the patches whose
        fraction of valid pixels (:attr:`valid`) is at least ``min_valid`` (0.5: half the
        patch), skipping NaN values. Both are per channel; a channel is NaN when nothing
        qualifies. There is no ``"sum"``: overlapping patches would count twice."""
        if how == "mean":
            # in row bands, so a large scene never needs its whole map at once
            total = np.zeros(self.values.shape[1], dtype=np.float64)
            count = np.zeros(self.values.shape[1], dtype=np.int64)
            band = max(1, 2**22 // max(1, self.shape[1]))
            for start in range(0, self.shape[0], band):
                stop = min(start + band, self.shape[0])
                part = self.to_map(rows=(start, stop))  # NaN at nodata and unpredicted pixels
                # per channel: a NaN the model gave in one channel leaves the others alone
                keep = ~np.isnan(part)
                total += np.where(keep, part, 0).sum(axis=(1, 2), dtype=np.float64)
                count += keep.sum(axis=(1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(count > 0, total / count, np.nan).astype(np.float32)
        if how == "max":
            keep = self.valid >= min_valid
            if not keep.any():
                return np.full(self.values.shape[1], np.nan, dtype=np.float32)
            # fmax skips NaN (a channel is NaN only if every patch's is), as "mean" does
            return np.fmax.reduce(self.values[keep], axis=0).astype(np.float32)
        raise ValueError(
            f"reduce supports 'mean' and 'max', got {how!r} (no 'sum': overlapping patches "
            f"would be counted twice)."
        )
