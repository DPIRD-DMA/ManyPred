"""Tiled mode: one scene -> patches -> runner -> blended mosaic."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .batch import _factor_of, build_normalizers, one_raster_spec, unwrap
from .blend import BandMosaic, Mosaic, MosaicBase, check_merge, gradient_weights
from .collate import SINGLE, as_inputs
from .devices import device_lock, resolve_device, resolve_dtype, warn_caller
from .inputs import (
    InputInfo,
    check_channels_first,
    describe_inputs,
    scale_window,
    window_multiple,
)
from .lazy import PackedMask, decimated, is_reader, read_window
from .lazy import nodata_fraction as reader_nodata_fraction
from .lazy import nodata_mask as reader_nodata_mask
from .normalize import has_scene_scope, resolve_scene
from .outputs import (
    FINISH_CHUNK,
    OutputSpec,
    TilePredictions,
    check_argmax,
    per_output,
)
from .outputs import mask_nodata as mask_invalid
from .runner import (
    Batch,
    Prediction,
    RawBatch,
    Runner,
    cast_input,
    check_outputs,
    ensemble,
    program_patch_size,
    set_up_run,
    warn_not_classes,
    whole,
)
from .spec import (
    fit_patch,
    input_order,
    input_scales,
    named_output_factors,
    output_alias,
    output_factors,
)
from .windows import (
    Window,
    adjust_patch_size,
    nodata_fraction,
    nodata_pixels,
    plan_windows,
    window_block,
)


def _check_scene(image: Any) -> None:
    if len(image.shape) != 3:
        hint = {
            2: " Pass a single-band (H, W) image as image[None].",
            4: " Pass a (T, C, H, W) time series in a dict, {name: series}, named as the "
            "model's tiling_spec() names it.",
        }.get(len(image.shape), "")
        raise ValueError(f"The image must be (C, H, W), found shape {tuple(image.shape)}.{hint}")
    check_channels_first(tuple(image.shape), "The image")


def read_patch(image: Any, window: Window, nodata: float | None) -> Any:
    """The image (array, tensor or lazy reader) under a window; parts past the scene edge are
    filled with :func:`pad_value`."""
    top, bottom, left, right = window
    height, width = image.shape[-2:]
    rows, cols = min(bottom, height) - top, min(right, width) - left
    shape = (*image.shape[:-2], bottom - top, right - left)
    if is_reader(image):
        inside = read_window(image, (top, top + rows, left, left + cols))
        if (rows, cols) == shape[-2:]:
            return inside
        image, top, left = inside, 0, 0
    elif bottom <= height and right <= width:
        return image[..., top:bottom, left:right]
    pad = pad_value(image, nodata)  # from the data read, whatever a reader declares
    if isinstance(image, torch.Tensor):
        patch = torch.full(shape, pad, dtype=image.dtype, device=image.device)
    else:
        patch = np.full(shape, pad, dtype=image.dtype)
    patch[..., :rows, :cols] = image[..., top : top + rows, left : left + cols]
    return patch


def pad_value(image: Any, nodata: float | None) -> float:
    """Nodata where the image (an array or tensor) dtype can hold it exactly, else 0."""
    if nodata is None:
        return 0
    if isinstance(image, torch.Tensor):
        floating = image.is_floating_point()
        boolean = image.dtype == torch.bool
        info = None if floating or boolean else torch.iinfo(image.dtype)
    else:
        dtype = np.asarray(image).dtype if not hasattr(image, "dtype") else np.dtype(image.dtype)
        floating = np.issubdtype(dtype, np.floating)
        boolean = dtype == np.bool_
        info = None if floating or boolean else np.iinfo(dtype)
    if floating:
        return nodata
    if boolean:
        return bool(nodata) if nodata in (0, 1) else 0
    if math.isnan(nodata) or nodata != int(nodata) or not info.min <= nodata <= info.max:
        return 0
    return int(nodata)


def iter_patch_batches(
    inputs: dict[str, Any],
    infos: dict[str, InputInfo],
    windows: list[Window],
    batch_size: int,
    nodata: float | None,
) -> Iterator[RawBatch]:
    """Batches of patch views per input (collation copies them into one buffer): each tiled
    input under its scaled window (padded past the edge with nodata where its dtype can hold
    it), vectors repeated."""
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        patches = {}
        for name, x in inputs.items():
            info = infos[name]
            if not info.tiled:
                patches[name] = [x] * len(chunk)
            else:
                patches[name] = [read_patch(x, scale_window(w, info.scale), nodata) for w in chunk]
        yield RawBatch(patches, [None] * len(chunk), chunk)


@dataclass(frozen=True)
class Layout:
    """How one output tiles: ``image`` (one (C,) vector per patch) or dense with ``factor`` f
    (a patch of P gives P / f pixels)."""

    channels: int
    image: bool = False
    factor: int = 1


class AmbiguousFactor(ValueError):
    """A dense output's size fits several resolutions for the patch it came from (a 2 x 2
    output of a 5 x 5 patch is 1/3 or 1/4 of it)."""


def layout_of(
    name: str,
    x: torch.Tensor,
    patch_size: int,
    declared: tuple[int, ...] = (),
    full: bool = True,
    expected: int | None = None,
) -> Layout:
    """The layout of an output from its shape for one ``patch_size`` patch. An output of h
    pixels has a factor f with ceil(P / f) = h. When several f fit, it is, in order: the one
    among the ``declared`` factors (the tiling spec's); P / h for the ``full`` patch size,
    which tiling needs to be a multiple of every output's factor (a patch shrunk to a small
    scene need not be); else :class:`AmbiguousFactor` is raised. ``expected``: the factor the
    spec declares for this output, which must fit (ValueError otherwise, as in batch mode)."""
    label = f"Output {name!r}" if name else "The model output"
    if x.ndim == 2:
        return Layout(x.shape[1], image=True)
    height, width = x.shape[-2:]
    factor = -(-patch_size // height)
    if -(-patch_size // factor) != height or height != width:
        raise ValueError(
            f"{label} is {height} x {width} for a {patch_size} x {patch_size} patch; dense "
            f"outputs must be the patch size, or ceil(P / f) for one integer f."
        )
    # ceil(P / f) = h for every f in [ceil(P / h), floor((P - 1) / (h - 1))]
    largest = math.inf if height == 1 else (patch_size - 1) // (height - 1)
    if expected is not None:
        if not factor <= expected <= largest:
            raise ValueError(
                f"{label} is {height} x {width} for a {patch_size} x {patch_size} patch, but "
                f"the tiling spec declares it at 1/{expected} of the input, which would be "
                f"{-(-patch_size // expected)} x {-(-patch_size // expected)}. Check the "
                f"output's scale in the model's tiling_spec() (or spec=)."
            )
        return Layout(x.shape[1], factor=expected)
    if largest > factor:
        fits = {f for f in declared if factor <= f <= largest}
        if len(fits) != 1 and full and patch_size % height == 0:
            fits = {patch_size // height}
        if len(fits) != 1:
            top = "any larger" if height == 1 else f"up to 1/{largest}"
            raise AmbiguousFactor(
                f"{label} is {height} x {width} for a {patch_size} x {patch_size} patch, which "
                f"fits a resolution of 1/{factor} of the input or {top}. Declare its scale "
                f"in the tiling spec (spec={{'outputs': {{{name!r}: {{'scale': 1 / f}}}}}}), "
                f"or use a patch_size that is a multiple of its factor."
            )
        factor = fits.pop()
    return Layout(x.shape[1], factor=factor)


class Replan(Exception):  # noqa: N818 - control flow, not an error
    """The first batch showed outputs at a resolution the windows are not aligned to."""

    def __init__(self, layouts: dict[str, Layout]):
        super().__init__("re-plan windows")
        self.layouts = layouts


class MosaicSink:
    """Blends dense outputs into one :class:`Mosaic` each (on that output's grid, merged as
    ``merge`` says per output) and collects image-level outputs per patch. Output layouts come
    from the first batch (or ``layouts`` when already known); if an output's factor does not
    divide ``multiple``, the windows are not aligned to its grid and :class:`Replan` is
    raised."""

    mosaic_class: type[MosaicBase] = Mosaic

    def __init__(
        self,
        shape: tuple[int, int],
        patch_size: int,
        overlap: int,
        multiple: int,
        device: torch.device,
        to_host: bool,
        pbar: tqdm,
        layouts: dict[str, Layout] | None = None,
        dtype: torch.dtype = torch.float32,
        auto_device: bool = False,
        inference_device: torch.device | None = None,
        merge: Any = "blend",
        class_specs: Callable[[], dict[str, OutputSpec] | None] | None = None,
        declared: tuple[int, ...] = (),
        full_patch: bool = True,
        alias: str | None = None,
        named_factors: Mapping[str, int] | None = None,
    ):
        self.shape = shape
        self.alias = alias  # the spec's name for a bare tensor's head, as merge may call it
        self.declared = declared  # output factors the tiling spec declares
        self.named_factors = named_factors  # {output: the factor the spec declares for it}
        self.full_patch = full_patch  # the patch size asked for, not one shrunk to the scene
        self.merge = merge
        # given when nothing will ask for merged values: argmax outputs then need no weights
        self.class_specs = class_specs
        self.merges: dict[str, str] = {}
        self.dtype = dtype
        # mosaic_device="auto": the CPU if the mosaics wouldn't fit on the GPU, decided when
        # their sizes are known; ``moved`` is then (bytes needed, bytes free)
        self.auto_device = auto_device and self.mosaic_class is Mosaic and device.type == "cuda"
        self.inference_device = inference_device or device
        self.moved: tuple[int, int] | None = None
        self.patch_size, self.overlap, self.multiple = patch_size, overlap, multiple
        self.device = device
        self.to_host = to_host
        self.pbar = pbar
        self.layouts: dict[str, Layout] | None = None
        self.mosaics: dict[str, MosaicBase] = {}
        self.tiles: dict[str, list[np.ndarray]] = {}
        self.windows: list[Window] = []
        if layouts is not None:
            self._start(layouts)

    def _start(self, layouts: dict[str, Layout], aligned: bool = False) -> None:
        if not aligned and any(self.multiple % layout.factor for layout in layouts.values()):
            raise Replan(layouts)
        self.layouts = layouts
        self.merges = per_output(self.merge, list(layouts), "merge", self.alias)
        for name, layout in layouts.items():
            if layout.image and isinstance(self.merge, Mapping) and self.merges[name] == "cut":
                raise ValueError(
                    f"merge='cut' was given for image-level output {name!r}, which has one "
                    f"value per patch (TilePredictions); merge applies to dense outputs."
                )
            self.merges[name] = self.merges[name] or "blend"  # not named in a dict
        if self.auto_device:
            self.auto_device = False
            self._choose_device(layouts)
        for name, layout in layouts.items():
            if layout.image:
                self.tiles[name] = []
                continue
            self.mosaics[name] = self._mosaic(name, self.dtype)

    def _choose_device(self, layouts: dict[str, Layout]) -> None:
        """Hold the mosaics on the CPU if they, plus finishing's temporaries for the largest
        output, would take more than 80% of the GPU's free memory (counting PyTorch's cached
        free blocks)."""
        free, _ = torch.cuda.mem_get_info(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        size = torch.empty((), dtype=self.dtype).element_size()
        need = largest = 0
        for name, layout in layouts.items():
            if layout.image:
                continue
            pixels = -(-self.shape[0] // layout.factor) * -(-self.shape[1] // layout.factor)
            if self._sums_only(name):  # no weights, and argmax in pieces into uint8
                need += layout.channels * pixels * size
                largest = max(largest, pixels + FINISH_CHUNK * 8)
            else:
                need += (layout.channels + 1) * pixels * size
                specs = self.class_specs() if self.class_specs is not None else None
                piecewise = bool(specs) and _as_values(specs[name])  # widened on the way over
                largest = max(
                    largest, FINISH_CHUNK * 4 if piecewise else 2 * layout.channels * pixels * 4
                )
        if need + largest > 0.8 * free:
            self.moved = (need + largest, free)
            self.device = torch.device("cpu")
            self.to_host = self.inference_device.type == "cuda"

    def _mosaic(self, name: str, dtype: torch.dtype) -> MosaicBase:
        assert self.layouts is not None
        layout = self.layouts[name]
        f = layout.factor
        return self.mosaic_class(
            layout.channels,
            -(-self.shape[0] // f),
            -(-self.shape[1] // f),
            gradient_weights(self.patch_size // f, self.overlap // f),
            self.device,
            dtype,
            merge=self.merges[name],
            **({"track_weight": False} if self._sums_only(name) else {}),
        )

    def _sums_only(self, name: str) -> bool:
        """Whether this output's mosaic needs only the sums: argmax of a blend (in a
        :class:`Mosaic`; a :class:`BandMosaic` always keeps the weights)."""
        if self.mosaic_class is not Mosaic:
            return False
        specs = self.class_specs() if self.class_specs is not None else None
        return bool(specs) and specs[name].is_class and self.merges[name] == "blend"

    def __call__(self, prediction: Prediction) -> None:
        if self.layouts is None:
            self._start(
                {
                    n: layout_of(
                        n,
                        x,
                        self.patch_size,
                        self.declared,
                        self.full_patch,
                        _factor_of(self.named_factors, n),
                    )
                    for n, x in prediction.outputs.items()
                }
            )
        assert self.layouts is not None
        windows = prediction.items
        for name, x in prediction.outputs.items():
            layout = self.layouts[name]
            if layout.image:
                self.tiles[name].append(x.numpy(force=True))
                continue
            f = layout.factor
            scaled = [(t // f, -(-b // f), lft // f, -(-r // f)) for t, b, lft, r in windows]
            self.mosaics[name].add(x, scaled)
        self.windows.extend(windows)
        self.pbar.update(len(windows))


def warn_if_overflowed(sums: torch.Tensor, name: str) -> None:
    """Warn when a float16 mosaic's sums aren't finite. An overflow gives infinity, or NaN once
    opposite infinities meet, and the model may have returned NaN too, so this can't tell them
    apart."""
    if sums.dtype != torch.float16 or sums.numel() == 0:
        return
    # aminmax passes NaN on and makes no temporaries, so the usual case costs no memory
    if torch.isfinite(torch.stack(torch.aminmax(sums))).all():
        return
    label = f"Output {name!r}" if name else "The output"
    warn_caller(
        f"{label} has infinite or NaN sums in its float16 mosaic: values summed past 65504, or "
        f"the model returned them. Pass mosaic_dtype='float32'."
    )


def resolve_mosaic_device(
    mosaic_device: str | torch.device | None, device: torch.device
) -> torch.device | None:
    """``"auto"`` -> None (chosen per scene), None -> the inference device, else a device."""
    if isinstance(mosaic_device, str) and mosaic_device == "auto":
        return None
    return resolve_device(mosaic_device) if mosaic_device is not None else device


def _on_grid(data: torch.Tensor, factor: int) -> torch.Tensor:
    """A (h, w) mask of input pixels on an output's grid: at reduced resolution, True where
    any pixel of the block is."""
    if factor == 1:
        return data
    pooled = F.max_pool2d(data[None, None].float(), factor, stride=factor, ceil_mode=True)
    return pooled[0, 0] > 0


def finish(
    mosaic: Mosaic,
    spec: OutputSpec,
    data: torch.Tensor,
    factor: int,
    mask: bool = True,
    consume: bool = False,
) -> np.ndarray:
    """Final (C, h, w) array of a dense output: argmax of the weighted sums (as OmniCloudMask
    does, so pixels without a prediction are class 0, and any NaN came from the model; of the
    values with ``merge="cut"``), else the merged values through ``spec``; then, with ``mask``,
    masked where the (H, W) ``data`` is False. ``consume``: the mosaic may be used up, which
    saves memory."""
    warn_if_overflowed(mosaic.sum, spec.name)
    if spec.is_class:
        # on the sums as they are and masked on the host, so finishing adds next to nothing to
        # the device's memory
        data = _on_grid(data, factor)
        check_argmax(mosaic.sum[None], data, spec.name)
        classes = spec.finalize(mosaic.sum[None], class_dim=1)[0].numpy(force=True)
        if mask:
            classes[:, ~data.numpy()] = 0
        return classes
    if consume and _as_values(spec):
        return _values_on_host(mosaic.take_blended(), _on_grid(data, factor) if mask else None)
    blended = (mosaic.take_blended().float() if consume else mosaic.blended())[None]
    out = spec.finalize(blended, class_dim=1)[0]
    del blended  # not held while masking what a callable made from it
    if mask:
        out = mask_invalid(out, _on_grid(data.to(out.device), factor))
    return out.numpy(force=True)


def _values_on_host(values: torch.Tensor, data: torch.Tensor | None) -> np.ndarray:
    """(C, H, W) values, used up, as a float32 host array: widened and masked (NaN where the
    host's (H, W) ``data`` is False) a few rows at a time on the way over, so the device never
    holds a float32 copy of the mosaic."""
    channels, height, width = values.shape
    out = np.empty((channels, height, width), dtype=np.float32)
    rows = max(1, FINISH_CHUNK // max(1, channels * width))
    for top in range(0, height, rows):
        chunk = values[:, top : top + rows].float()
        if data is not None:
            chunk = chunk.masked_fill_(~data[top : top + rows].to(chunk.device), torch.nan)
        torch.from_numpy(out[:, top : top + rows]).copy_(chunk)
    return out


def tile_predictions(
    tiles: list[np.ndarray],
    channels: int,
    spec: OutputSpec,
    windows: list[Window],
    nodata: np.ndarray,
    patch_size: int,
    overlap: int,
) -> TilePredictions:
    if spec.mode not in ("logits", "probs") and not spec.named:
        # a mode given for all outputs (e.g. argmax for a multi-head model) applies to the
        # dense ones; image-level heads keep their per-patch values
        warn_caller(
            f"output={spec.mode!r} does not apply to image-level output {spec.name!r} in tiled "
            f"mode; it gives per-patch values (TilePredictions). Set output per output to "
            f"silence this."
        )
    elif spec.mode not in ("logits", "probs"):
        raise ValueError(
            f"Image-level output {spec.name!r} gives per-patch values in tiled mode; use "
            f"output='logits' or 'probs' and reduce them (TilePredictions.reduce / to_map)."
        )
    height, width = nodata.shape
    clipped = np.array(
        [(t, min(b, height), lft, min(r, width)) for t, b, lft, r in windows], dtype=np.int64
    ).reshape(-1, 4)
    # fraction of the whole patch: padding past the scene edge counts as invalid
    valid = np.array(
        [
            (block.size - np.count_nonzero(block)) / patch_size**2
            for block in (window_block(nodata, tuple(w)) for w in clipped)
        ],
        dtype=np.float32,
    )
    values = np.concatenate(tiles) if tiles else np.zeros((0, channels), dtype=np.float32)
    return TilePredictions(
        values=values.astype(np.float32, copy=False),
        windows=clipped,
        valid=valid,
        shape=(height, width),
        weights=gradient_weights(patch_size, overlap).numpy(),
        valid_map=_invert(nodata) if nodata.any() else None,
    )


def predict_array(
    model: Any,
    image: Any,
    *,
    patch_size: int = 1000,
    overlap: int = 300,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    mosaic_device: str | torch.device | None = "auto",
    mosaic_dtype: str | torch.dtype = "float32",
    shift_patches: bool = True,
    adapt_patch_size: bool = False,
    dtype: str | torch.dtype = "float32",
    compile: bool = False,
    normalize: Any = None,
    activation: Any = None,
    output: Any = "auto",
    nodata: float | None = None,
    mask_nodata: bool = True,
    merge: str | dict[str, str] = "blend",
    postprocess: Callable[[Any], Any] | None = None,
    scales: dict[str, float] | None = None,
    spec: dict[str, Any] | None = None,
    progress: bool = True,
) -> Any:
    """Predict one scene in overlapping patches and blend them into a mosaic.

    Patches of ``patch_size`` are laid on a grid with stride ``patch_size - overlap``; the
    last row and column are moved back to the scene edge. Patches that are entirely nodata are
    skipped, and (``shift_patches``) patches with a nodata edge are shifted inward, as
    OmniCloudMask does. Predictions are weighted with
    :func:`~manypred.blend.gradient_weights`, summed (in ``mosaic_dtype``) and divided by the
    summed weights. A patch larger than the scene shrinks to fit, with a warning, unless the
    model has a fixed input size: then the patch is padded past the scene edge (with
    ``nodata``) and the output cropped.

    Args:
        model: An ``nn.Module`` or callable taking a (B, C, h, w) tensor and returning a tensor
            or a dict of tensors, or a list of them (an ensemble: raw outputs are averaged).
            Modules are moved to ``device`` and ``dtype`` in place and set to eval mode. A
            callable that isn't a module may declare ``device`` (the default device),
            ``batch_size`` (replaces ``batch_size``; short batches are padded by repeating
            their last item) and ``fixed_size`` (sets ``patch_size``; it must be square to tile),
            as a ``QuantizedProgram`` from ``manyseg.load`` does.
        image: (C, H, W) numpy array or tensor (CPU or on the device), or a dict of inputs
            for a multi-input model: rasters (C, h, w) at any resolution ratio to the first
            raster (the primary, which sets the patch grid and the nodata mask), time series
            (T, C, h, w) and vectors (D,); see :mod:`manypred.inputs`.
        patch_size: Patch height and width in pixels.
        overlap: Pixels shared by neighbouring patches. The blend weights ramp over at most
            ``patch_size // 2`` of them.
        batch_size: Patches per batch.
        device: Inference device; default CUDA, then MPS, then the CPU.
        mosaic_device: Device holding the blended mosaic. ``"auto"`` (the default): the
            inference device, unless on CUDA the mosaic (and what finishing it
            needs) would take more than 80% of the free GPU memory, in which case
            it is held on the CPU, with a warning (predictions are copied over on their own
            stream, so the GPU keeps working). None: always the inference device. Or a device
            such as ``"cpu"``.
        mosaic_dtype: Dtype the mosaic is summed in: ``"float32"`` (the default), or ``"float16"``
            or ``"bfloat16"``, which take half the memory (a full Sentinel-2 tile with 4 classes
            takes 1.2 GB instead of 2.4 GB) and lose precision. Float16 sums overflow past 65504,
            with a warning.
        shift_patches: Move patches whose edge row or column is entirely nodata inward until
            it isn't (OmniCloudMask's behaviour), so edge patches see more valid data. False
            keeps the plain grid, the same for every scene of a size.
        adapt_patch_size: When over 30 % of the scene is nodata and the patch is over half
            the scene, shrink the patch to half the scene (OmniCloudMask's behaviour; it
            changes the model's context from scene to scene). Default False.
        dtype: Inference dtype: ``"float32"`` (the default), ``"float64"`` (not on MPS), or
            ``"float16"`` or ``"bfloat16"``,
            which are faster and use less GPU memory on hardware that supports them but round
            the model's weights and outputs: outputs in large units lose the most (values around
            1500 can be off by units). The weights of an ``nn.Module`` stay rounded after the call.
        compile: ``torch.compile`` the models, with static shapes (shorter batches are padded
            to the first batch's size; outputs must not depend on the other patches in a batch,
            which batching assumes anyway).
        normalize: Normalisation spec (:mod:`manypred.normalize`); ``scope="tile"`` statistics
            are per patch, and ``scope="scene"`` statistics use every pixel (with a lazy reader, a
            regular sample of at most 4 million pixels). None uses the model's stored
            normalisation (``tiling_spec()``) if it has one, else no steps; False always means no
            steps. Either way values arrive as float32, with missing ones (``nodata``, NaN,
            infinity) as 0.
        activation: Applied to each patch before blending; see :mod:`manypred.outputs`.
        output: ``"auto"`` (the default: ``"probs"`` for outputs with an activation, given
            or declared in the model's ``tiling_spec()``, else ``"logits"``), ``"logits"``,
            ``"probs"``, ``"argmax"`` or a callable
            on the blended (1, C, H, W) float32 mosaic. ``"argmax"`` raises a RuntimeError if
            the model returned NaN at a pixel with data, which class 0 would otherwise hide.
        nodata: Input value marking missing data (NaN allowed), or None (the default: no
            value marks missing data; NaN and infinity always do). The model sees missing
            values as 0.
            Satellite scenes with zero-filled borders want ``nodata=0``.
        mask_nodata: Mark nodata pixels in outputs: NaN in float outputs, 0 in class outputs.
            Without it, nodata pixels a patch covered keep what the model returned. Pixels no
            patch covered are NaN in float outputs and class 0 either way.
        merge: How overlapping patches are merged, for all dense outputs or per output
            (``{name: merge}``; outputs not named blend). Image-level outputs keep one value
            per patch and are never merged, so a single value leaves them alone, and naming one
            with ``"cut"`` in a dict is an error. ``"blend"`` (the default): the
            gradient-weighted mean. ``"cut"``: each pixel takes the values (all channels) of the
            patch whose gradient weight is largest there, the patch placed first on equal
            weights, so every pixel comes from the most central patch covering it and nothing
            is averaged. For outputs that are decoded into vectors or instances afterwards,
            where averaging two patches' views of an edge blurs it. Same memory as blending.
        scales: ``{input_name: ratio}`` of a raster's resolution to the primary input's,
            when it can't be inferred from the shapes (e.g. rounded sizes). Patches of the
            primary input map to ``patch_size * ratio`` pixels of that input, so the patch size
            and overlap must make those whole.
        spec: A tiling spec (:mod:`manypred.spec`); default ``model.tiling_spec()`` if the
            model has one. It fills in input scales, output resolutions, patch-size limits
            and, when ``normalize`` is None, the stored normalisation.
        postprocess: ``f(result) -> anything``, run once on the finished result (the array,
            or the dict of arrays): for derived outputs that must not be blended, such as
            instances, peaks or counts.
        progress: Show a progress bar.

    Outputs at reduced resolution ((B, C, ceil(P/f), ceil(P/f)) per patch) are blended on
    their own ceil(H/f) x ceil(W/f) grid; then ``patch_size`` and ``overlap`` must be
    divisible by f, windows start at multiples of f, and patches at the far edges may be
    padded with ``nodata`` (as the model pads its own input). Image-level outputs (B, C) give
    a :class:`~manypred.TilePredictions`; they need an activation, passed or declared by the
    model, or ``output="logits"``.

    Returns:
        (C, H, W) array (``(1, H, W)`` uint8 for ``"argmax"``, int64 past 256 classes;
        ``(C, ceil(H/f), ceil(W/f))``
        for a reduced-resolution output; :class:`~manypred.TilePredictions` for an
        image-level one), a dict of them for a dict-output model, or what ``postprocess``
        returns.
    """
    patch_size, overlap = whole("patch_size", patch_size), whole("overlap", overlap)
    batch_size = whole("batch_size", batch_size, minimum=1)
    setup = set_up_run(
        model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        default_batch_size=1,
        spec=spec,
        normalize=normalize,
        activation=activation,
        output=output,
        nodata=nodata,
        compile=compile,
    )
    runner, tiling = setup.runner, setup.tiling
    patch_size, overlap = fit_patch(tiling, patch_size, overlap)
    patch_size, overlap = program_patch_size(runner.models, patch_size, overlap, adapt_patch_size)
    tiler = Tiler(
        runner=runner,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=setup.batch_size,
        mosaic_device=resolve_mosaic_device(mosaic_device, runner.device),
        mask_nodata=mask_nodata,
        progress=progress,
        normalize=setup.normalize,
        scales=scales if scales is not None else (input_scales(tiling) or None),
        factors=tuple(output_factors(tiling)),
        named_factors=named_output_factors(tiling),
        order=tuple(input_order(tiling)),
        mosaic_dtype=mosaic_dtype,
        shift_patches=shift_patches,
        adapt_patch_size=adapt_patch_size,
        merge=merge,
        alias=output_alias(tiling),
    )
    result, _ = tiler.predict(image)
    result = unwrap(result)
    return postprocess(result) if postprocess is not None else result


@dataclass
class Tiler:
    """Predicts whole scenes with one :class:`Runner` (models prepared once)."""

    runner: Runner
    patch_size: int
    overlap: int
    batch_size: int
    mosaic_device: torch.device | None  # None: chosen per scene ("auto")
    mask_nodata: bool
    progress: bool
    normalize: Any = None
    scales: dict[str, float] | None = None
    factors: tuple[int, ...] = ()  # output factors known in advance (tiling_spec)
    # {output: factor} the tiling spec declares (spec.named_output_factors), checked per output
    named_factors: dict[str, int] = field(default_factory=dict)
    order: tuple[str, ...] = ()  # input names in the model's spec order (its primary comes first)
    mosaic_dtype: str | torch.dtype = "float32"
    shift_patches: bool = True
    adapt_patch_size: bool = False
    merge: str | dict[str, str] = "blend"  # per output, see predict_array
    alias: str | None = None  # the spec's name for a single head returned as a bare tensor
    _layouts: dict[Any, dict[str, Layout]] = field(default_factory=dict)
    _warned_moved: bool = False  # the mosaic_device="auto" warning, once per Tiler
    # {output: factor, or None for an image-level one} of the last scene predicted
    last_factors: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_merge(self.merge)

    def probe(
        self, inputs: dict[str, Any], infos: dict[str, InputInfo], patch_size: int
    ) -> dict[str, Layout]:
        """Output layouts from one forward pass on a patch of zeros (vectors: the scene's
        values): for scenes without a single patch to predict, and for outputs whose
        resolution a patch shrunk to a small scene can't tell. If ``patch_size`` can't tell
        either, the full ``self.patch_size`` is tried."""
        key = _structure(inputs, infos)
        if key in self._layouts:
            return self._layouts[key]
        try:
            layouts = {
                name: layout_of(
                    name,
                    x,
                    patch_size,
                    self.factors,
                    patch_size == self.patch_size,
                    _factor_of(self.named_factors, name),
                )
                for name, x in self._probe_outputs(inputs, infos, patch_size).items()
            }
        except AmbiguousFactor:
            if patch_size >= self.patch_size:
                raise
            return self.probe(inputs, infos, self.patch_size)
        self._layouts[key] = layouts
        return layouts

    def _probe_outputs(
        self, inputs: dict[str, Any], infos: dict[str, InputInfo], patch_size: int
    ) -> dict[str, torch.Tensor]:
        """The models' outputs for one ``patch_size`` patch of zeros."""
        runner = self.runner
        example = {}
        for name, x in inputs.items():
            info = infos[name]
            if info.tiled:
                size = int(patch_size * info.scale)
                shape = (1, *x.shape[:-2], size, size)
                example[name] = torch.zeros(shape, dtype=runner.dtype, device=runner.device)
            else:
                vector = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
                vector = vector.to(runner.device)[None]
                if vector.is_floating_point():  # float32 first, as a batch's vectors are
                    vector = vector.float()
                example[name] = cast_input(name, vector, inputs, runner.dtype, runner.exact_vectors)
        with runner.claim(), device_lock(runner.device):
            is_uint16 = dict.fromkeys(example, False)
            # compiled for the scenes' batches, not this one patch
            example_batch = Batch(dict(example), is_uint16, [None], [None])
            runner.prepare(example_batch, compile_size=self.batch_size)
            with torch.no_grad():
                model_in = example[SINGLE] if set(example) == {SINGLE} else example
                raw = ensemble(runner.models, model_in)
        check_outputs(raw, 1)
        return raw

    def _normalizers(self, inputs: dict[str, Any], infos: dict[str, InputInfo]) -> dict:
        """Normalisers for this scene's inputs; scene-scope statistics are computed here, once
        over the whole scene, before tiling."""
        names = [name for name, info in infos.items() if info.tiled]
        spec = one_raster_spec(self.normalize, names)
        if names == [SINGLE]:
            per_input = {SINGLE: spec}
        elif isinstance(spec, dict) and "type" not in spec:
            per_input = spec
        else:  # no spec, or one spec over several rasters (which build_normalizers rejects)
            return build_normalizers(spec, list(inputs), batch_mode=False)
        resolved = {}
        for name, steps in per_input.items():
            if has_scene_scope(steps) and name in inputs:
                x = inputs[name]
                if is_reader(x):
                    x = decimated(x)  # statistics from every k-th pixel, read in row bands
                # read in row chunks, in its own dtype (no float32 copy of the scene)
                scene = x if isinstance(x, torch.Tensor) else np.asarray(x)
                steps = resolve_scene(steps, scene, self.runner.nodata)
            resolved[name] = steps
        if set(resolved) == {SINGLE}:
            return build_normalizers(resolved[SINGLE], [SINGLE], batch_mode=False)
        return build_normalizers(resolved, list(inputs), batch_mode=False)

    def _run(self, image: Any, make_sink: Callable[..., MosaicSink]) -> _Run:
        """Plan the scene's windows and stream them through the runner into a sink from
        ``make_sink(**kwargs)``. Output scales are learnt from the first batch; windows start
        at multiples of every input's scale and of the coarsest output scale, re-planning
        once if the first batch shows they don't."""
        runner = self.runner
        inputs = {
            name: x if isinstance(x, torch.Tensor) or is_reader(x) else np.asarray(x)
            for name, x in as_inputs(image).items()
        }
        if set(inputs) == {SINGLE}:
            _check_scene(inputs[SINGLE])
        primary, infos = describe_inputs(inputs, self.scales, self.order)
        runner.primary = primary
        image = inputs[primary]
        height, width = image.shape[-2:]

        nodata_map = _nodata_map(image, runner.nodata)
        runner.normalizers = self._normalizers(inputs, infos)
        device, mosaic_device = runner.device, self.mosaic_device or runner.device
        key = _structure(inputs, infos)
        input_multiple = window_multiple(infos)
        # a model with a fixed input size takes only that size: pad patches past small scenes
        fixed = runner.fixed_size is not None and self.patch_size == runner.fixed_size[0]
        pbar = tqdm(
            desc=f"Predicting on {device.type} {str(runner.dtype).split('.')[-1]}",
            disable=not self.progress,
        )
        try:
            while True:
                known = self._layouts.get(key)
                multiple = math.lcm(
                    input_multiple,
                    *self.factors,
                    *(layout.factor for layout in (known or {}).values()),
                )
                if multiple > 1 and (self.patch_size % multiple or self.overlap % multiple):
                    raise ValueError(
                        f"Reduced-resolution inputs or outputs need windows on multiples of "
                        f"{multiple}, so patch_size and overlap must be divisible by "
                        f"{multiple}; got {self.patch_size} and {self.overlap}."
                    )
                patch_size, overlap = adjust_patch_size(
                    height,
                    width,
                    # a pass over the whole scene, only needed to adapt the patch size
                    _nodata_fraction(image, runner.nodata) if self.adapt_patch_size else 0.0,
                    self.patch_size,
                    self.overlap,
                    multiple,
                    adapt=self.adapt_patch_size,
                    fixed=fixed,
                )
                windows = plan_windows(
                    nodata_map,
                    patch_size,
                    overlap,
                    shift=self.shift_patches,
                    multiple=multiple,
                    pad=fixed,
                )
                pbar.reset(total=len(windows))
                sink = make_sink(
                    shape=(height, width),
                    patch_size=patch_size,
                    overlap=overlap,
                    multiple=multiple,
                    device=mosaic_device,
                    to_host=mosaic_device.type == "cpu" and device.type == "cuda",
                    auto_device=self.mosaic_device is None,
                    inference_device=device,
                    pbar=pbar,
                    layouts=known,
                    windows=windows,
                    nodata=nodata_map,
                    dtype=resolve_dtype(self.mosaic_dtype),
                    merge=self.merge,
                    declared=self.factors,
                    full_patch=patch_size == self.patch_size,
                    alias=self.alias,
                    named_factors=self.named_factors,
                )
                batches = iter_patch_batches(inputs, infos, windows, self.batch_size, runner.nodata)
                try:
                    runner.run(batches, sink)
                except Replan as replan:
                    self._layouts[key] = replan.layouts
                    continue
                except AmbiguousFactor:
                    # a patch shrunk to a small scene: learn the outputs at the full patch size
                    if patch_size >= self.patch_size:
                        raise
                    self.probe(inputs, infos, self.patch_size)
                    continue
                break
        finally:
            pbar.close()
        if sink.moved is not None and not self._warned_moved:
            self._warned_moved = True
            need, free = sink.moved
            warn_caller(
                f"The mosaic for this scene needs about {need / 2**30:.1f} GB with finishing, "
                f"and the GPU has {free / 2**30:.1f} GB free, so it is blended on the CPU. "
                f"Pass mosaic_device= to choose (e.g. mosaic_device=None for the GPU)."
            )
        layouts = sink.layouts
        if layouts is None:
            # no patch to predict (all nodata): learn the outputs from a zero patch
            layouts = self.probe(inputs, infos, patch_size)
            sink._start(layouts, aligned=True)  # no windows to align
        self._layouts[key] = layouts
        specs = runner.specs
        if specs is None:  # no patch reached the runner
            specs = runner.make_specs(list(layouts))
            warn_not_classes(specs)
        for name, layout in layouts.items():
            specs[name].check_shape(2 if layout.image else 4)
        return _Run(sink, layouts, specs, nodata_map, patch_size, overlap)

    def _tiles(self, run: _Run) -> dict[str, TilePredictions]:
        return {
            name: tile_predictions(
                run.sink.tiles[name],
                layout.channels,
                run.specs[name],
                run.sink.windows,
                run.nodata,
                run.patch_size,
                run.overlap,
            )
            for name, layout in run.layouts.items()
            if layout.image
        }

    def predict(self, image: Any) -> tuple[dict[str, Any], np.ndarray | None]:
        """{output_name: array or TilePredictions} for one scene, and its (H, W) valid mask
        (None without ``mask_nodata``)."""
        result, valid, _ = self.predict_fields(image)
        return result, valid

    def predict_fields(
        self,
        image: Any,
        select: Callable[[dict[str, int | None], str | None], list[str]] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None, dict[str, np.ndarray]]:
        """As :meth:`predict`, plus the stitched values of the outputs ``select({output:
        factor, or None for an image-level one}, alias)`` names (``alias``: the spec's name for
        a bare tensor's output): merged, activated and masked, before any output conversion
        (argmax, callables), as float32 (C, H, W) arrays (the results themselves for outputs
        returned as values)."""
        runner = self.runner

        def make_sink(*, windows: Any, nodata: Any, **kwargs: Any) -> MosaicSink:
            # without select, nothing asks for the merged values of argmax outputs
            return MosaicSink(**kwargs, class_specs=None if select else lambda: runner.specs)

        run = self._run(image, make_sink)
        self.last_factors = _factors(run.layouts)
        names = select(self.last_factors, self.alias) if select is not None else []
        data = ~_unpacked(run.nodata)
        valid = data if self.mask_nodata else None
        mask = torch.from_numpy(data)
        fields: dict[str, np.ndarray] = {}
        with device_lock(self.runner.device, run.sink.device):
            result: dict[str, Any] = dict(self._tiles(run))
            for name, layout in run.layouts.items():
                if layout.image:
                    continue
                spec, mosaic = run.specs[name], run.sink.mosaics[name]
                consume = name not in names or _as_values(spec)
                result[name] = finish(
                    mosaic, spec, mask, layout.factor, self.mask_nodata, consume=consume
                )
                if name in names:
                    if _as_values(spec):
                        fields[name] = result[name]
                    else:
                        values = mosaic.blended()
                        if self.mask_nodata:
                            on_grid = _on_grid(mask, layout.factor).to(values.device)
                            values = mask_invalid(values, on_grid)
                        fields[name] = values.numpy(force=True)
        return {name: result[name] for name in run.specs}, valid, fields

    def predict_blocks(
        self, image: Any, writer: Any, hook: Any = None
    ) -> dict[str, TilePredictions]:
        """Predict one scene, writing dense outputs to ``writer`` in row blocks as soon as no
        remaining patch can change them, so only a band of rows is ever held. ``writer.write(
        name, top, block, shape, valid, factor=f)`` gets rows [top, top + h) of output ``name``
        (full shape ``shape``, f times coarser than the scene) and their (h, w) valid mask (or
        None). Returns image-level outputs.

        ``hook`` (a :class:`~manypred.blocks.BlockStream`) also gets the same rows of the
        outputs its ``select(factors)`` names, as stitched values (see
        :meth:`predict_fields`): ``hook.rows(top, {name: (C, h, W) array})``."""
        runner = self.runner

        def make_sink(**kwargs: Any) -> BlockSink:
            return BlockSink(
                **kwargs,
                writer=writer,
                specs=lambda: runner.specs,
                mask=self.mask_nodata,
                hook=hook,
            )

        run = self._run(image, make_sink)
        self.last_factors = _factors(run.layouts)
        assert isinstance(run.sink, BlockSink)
        run.sink.specs = lambda: run.specs
        with device_lock(runner.device, run.sink.device):
            run.sink.flush(final=True)
        return self._tiles(run)


@dataclass
class _Run:
    sink: MosaicSink
    layouts: dict[str, Layout]
    specs: dict[str, OutputSpec]
    nodata: Any
    patch_size: int
    overlap: int


class BlockSink(MosaicSink):
    """A :class:`MosaicSink` holding only a band of rows per dense output: after each batch,
    rows above the topmost remaining window are final, so they are finished (divided,
    output mode, masked) and handed to ``writer``."""

    mosaic_class = BandMosaic

    def __init__(
        self,
        *,
        windows: list[Window],
        nodata: Any,
        writer: Any,
        specs: Callable[[], dict[str, OutputSpec] | None],
        mask: bool,
        hook: Any = None,
        **kwargs: Any,
    ):
        # suffix minimum of window tops: rows above it are final once a window is done
        self.final = [kwargs["shape"][0]] * (len(windows) + 1)
        for i in range(len(windows) - 1, -1, -1):
            self.final[i] = min(windows[i][0], self.final[i + 1])
        self.done = 0
        self.nodata_map = nodata
        self.writer = writer
        self.specs = specs
        self.mask = mask
        self.hook = hook
        super().__init__(**kwargs)

    def __call__(self, prediction: Prediction) -> None:
        super().__call__(prediction)
        self.done += len(prediction.items)
        self.flush()

    def flush(self, final: bool = False) -> None:
        specs = self.specs()
        if specs is None or self.layouts is None:
            return
        height, width = self.shape
        final = final or self.done >= len(self.final) - 1
        hook, hooked, hooked_top = self.hook, {}, 0
        if hook is not None and hook.names is None:
            hook.select(_factors(self.layouts), self.alias)
        for name, mosaic in self.mosaics.items():
            f = self.layouts[name].factor
            until = -(-height // f) if final else self.final[self.done] // f
            taken = mosaic.take(until)
            if taken is None:
                continue
            top, sums, weight = taken
            spec = specs[name]
            warn_if_overflowed(sums, spec.name)
            blended = sums if spec.is_class else mosaic.merged(sums, weight)
            if self.mask or spec.is_class:
                bottom = min((top + blended.shape[1]) * f, height)
                rows = np.asarray(self.nodata_map[top * f : bottom, :])
                valid_t = _on_grid(torch.from_numpy(~rows).to(blended.device), f)
            if spec.is_class:
                check_argmax(blended[None], valid_t, spec.name)
            out = spec.finalize(blended[None], class_dim=1)[0]
            if out.shape[-2:] != blended.shape[-2:]:
                raise ValueError(
                    f"Output {name!r}: a callable output must keep the spatial shape; on lazy "
                    f"scenes it runs on row blocks, so it must also be pixel-wise."
                )
            valid = None
            if self.mask:
                out = mask_invalid(out, valid_t.to(out.device))
                valid = valid_t.numpy(force=True)
            shape = (out.shape[0], -(-height // f), -(-width // f))
            block = out.numpy(force=True)
            self.writer.write(name, top, block, shape, valid, factor=f)
            if hook is not None and name in hook.names:  # full resolution (select checks)
                if not _as_values(spec):  # the values argmax or a callable were taken from
                    values = mosaic.merged(sums, weight)
                    block = (mask_invalid(values, valid_t) if self.mask else values).numpy(
                        force=True
                    )
                hooked[name], hooked_top = block, top
        if hooked:
            hook.rows(hooked_top, hooked)


def _factors(layouts: dict[str, Layout]) -> dict[str, int | None]:
    """{output: factor, or None for an image-level one}."""
    return {name: None if layout.image else layout.factor for name, layout in layouts.items()}


def _as_values(spec: OutputSpec) -> bool:
    """Whether an output is returned as its stitched values (no argmax or callable)."""
    return spec.mode in ("logits", "probs")


def _structure(inputs: dict[str, Any], infos: dict[str, InputInfo]) -> tuple:
    """What output layouts depend on: input names, kinds, scales and non-spatial shapes."""
    return tuple(
        (name, info.kind, info.scale, tuple(inputs[name].shape[: -2 if info.tiled else None]))
        for name, info in infos.items()
    )


def _nodata_map(image: Any, nodata: float | None) -> np.ndarray | PackedMask:
    if is_reader(image):
        return reader_nodata_mask(image, nodata)
    if image.ndim == 4:  # a time series: nodata when every frame and band is
        image = image.reshape(-1, *image.shape[-2:])
    return nodata_pixels(image, nodata)


def _nodata_fraction(image: Any, nodata: float | None) -> float:
    if is_reader(image):
        return reader_nodata_fraction(image, nodata)
    return nodata_fraction(image, nodata)


def _unpacked(mask: np.ndarray | PackedMask) -> np.ndarray:
    return mask.unpacked() if isinstance(mask, PackedMask) else mask


def _invert(mask: np.ndarray | PackedMask) -> np.ndarray | PackedMask:
    """The valid map from a nodata map, staying bit-packed for lazy scenes."""
    if isinstance(mask, PackedMask):
        return PackedMask(~mask.packed, mask.shape[1])
    return ~mask
