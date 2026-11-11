"""Batch mode: many same-size chips, each predicted whole."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice, repeat
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .collate import SINGLE, as_inputs
from .geo import _factor
from .inputs import check_channels_first
from .io_lock import IO_LOCK
from .lazy import read_readers
from .normalize import NormalizeSpec, build_normalizer, steps_of
from .outputs import OutputSpec, check_argmax, mask_nodata
from .runner import (
    Prediction,
    RawBatch,
    set_up_run,
    whole,
)
from .spec import input_order, named_output_factors

_ITEM_NOTE = "While processing "


def name_item(error: Exception, item: Any) -> None:
    """Add the item to an error's traceback, keeping its type, unless it names one already (a
    save's error may surface while a later item is running)."""
    if not any(note.startswith(_ITEM_NOTE) for note in getattr(error, "__notes__", [])):
        error.add_note(f"{_ITEM_NOTE}{item}")


def load_item(
    item: Any,
    load_func: Callable | None,
    io_lock: bool = True,
    whole: bool = False,
    order: Sequence[str] = (),
) -> tuple[dict[str, Any], Any]:
    """Load an item with ``load_func`` if given, taking turns with saves unless ``io_lock`` is
    False. Returns ({input_name: image}, meta). With ``whole`` (batch mode), lazy readers are
    read whole, and ``meta`` defaults to the primary one's profile (see
    :func:`~manypred.lazy.read_readers`)."""
    meta = None
    if load_func is None:
        return as_inputs(item), meta
    try:
        with IO_LOCK.read() if io_lock else nullcontext():
            loaded = load_func(item)
        if isinstance(loaded, tuple):
            if len(loaded) != 2:
                raise ValueError(
                    f"load_func must return an image or an (image, meta) tuple, got a tuple "
                    f"of {len(loaded)}."
                )
            loaded, meta = loaded
        inputs = as_inputs(loaded)
        if whole:
            with IO_LOCK.read() if io_lock else nullcontext():
                inputs, meta = read_readers(inputs, meta, order)
        return inputs, meta
    except Exception as error:
        name_item(error, item)
        raise


def raster_names(batch: RawBatch) -> list[str]:
    """The inputs whose chips are (C, H, W) rasters or (T, C, H, W) time series."""
    return [
        name
        for name, images in batch.inputs.items()
        if (images.ndim - 1 if hasattr(images, "ndim") else len(images[0].shape)) in (3, 4)
    ]


def primary_input(batch: RawBatch, order: Sequence[str] = ()) -> str:
    """The first raster or time series in ``order`` (the model's spec), else in the batch."""
    rasters = raster_names(batch)
    if not rasters:
        raise ValueError("At least one input must be (C, H, W) raster or (T, C, H, W) chips.")
    return next((name for name in order if name in rasters), rasters[0])


def _shapes(inputs: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {name: tuple(image.shape) for name, image in inputs.items()}


def _check_image(shapes: dict[str, tuple[int, ...]]) -> None:
    if SINGLE in shapes and len(shapes[SINGLE]) != 3:
        hint = {
            2: " Pass a single-band (H, W) image as image[None].",
            4: " Pass a (T, C, H, W) time series in a dict, {name: series}.",
        }.get(len(shapes[SINGLE]), "")
        raise ValueError(f"Each image must be (C, H, W), found shape {shapes[SINGLE]}.{hint}")
    if SINGLE in shapes:
        check_channels_first(shapes[SINGLE], "Each image")


def _chunks(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def iter_item_batches(
    data: Iterable[Any],
    load_func: Callable | None,
    batch_size: int,
    shape_hint: str = "Pass group_by_shape=True to batch each size separately.",
    io_lock: bool = True,
    order: Sequence[str] = (),
) -> Iterator[RawBatch]:
    """Load items one batch at a time, the items of a batch in parallel threads, in input
    order. Every image must match the shapes of the first."""
    expected: dict[str, tuple[int, ...]] | None = None
    index = 0
    workers = max(1, min(batch_size, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk in _chunks(data, batch_size):
            loaded = list(
                executor.map(
                    load_item,
                    chunk,
                    repeat(load_func),
                    repeat(io_lock),
                    repeat(True),
                    repeat(order),
                )
            )
            for item, (inputs, _) in zip(chunk, loaded, strict=True):
                try:
                    shapes = _shapes(inputs)
                    if expected is None:
                        _check_image(shapes)
                        expected = shapes
                    elif shapes != expected:
                        raise ValueError(
                            f"All images must have the same shape. Image {index} has shape "
                            f"{_describe(shapes)}, the first had {_describe(expected)}. "
                            f"{shape_hint}"
                        )
                except Exception as error:
                    if load_func is not None:  # otherwise the item is the image itself
                        name_item(error, item)
                    raise
                index += 1
            names = list(loaded[0][0])
            yield RawBatch(
                inputs={name: [inputs[name] for inputs, _ in loaded] for name in names},
                metas=[meta for _, meta in loaded],
                items=chunk,
            )


def iter_grouped_batches(
    data: Iterable[Any], load_func: Callable | None, batch_size: int, order: Sequence[str] = ()
) -> Iterator[RawBatch]:
    """Like :func:`iter_item_batches`, but items of different shapes go to separate buckets:
    a bucket is yielded when it holds ``batch_size`` items, and partial buckets at the end.
    ``items`` are ``(index, item)`` pairs."""
    buckets: dict[tuple, list[tuple[int, Any, dict[str, Any], Any]]] = {}
    # with many distinct shapes buckets rarely fill; past this many waiting items the fullest
    # bucket goes as a partial batch, so loaded chips can't pile up
    max_waiting = 4 * batch_size

    def batch_of(bucket: list) -> RawBatch:
        names = list(bucket[0][2])
        return RawBatch(
            inputs={name: [inputs[name] for _, _, inputs, _ in bucket] for name in names},
            metas=[meta for _, _, _, meta in bucket],
            items=[(index, item) for index, item, _, _ in bucket],
        )

    workers = max(1, min(batch_size, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk in _chunks(enumerate(data), batch_size):
            loaded = executor.map(
                load_item,
                [item for _, item in chunk],
                repeat(load_func),
                repeat(True),
                repeat(True),
                repeat(order),
            )
            for (index, item), (inputs, meta) in zip(chunk, loaded, strict=True):
                shapes = _shapes(inputs)
                key = tuple(sorted(shapes.items()))
                if key not in buckets:
                    _check_image(shapes)
                    buckets[key] = []
                buckets[key].append((index, item, inputs, meta))
                if len(buckets[key]) == batch_size:
                    yield batch_of(buckets[key])
                    buckets[key] = []
                elif sum(len(b) for b in buckets.values()) >= max_waiting:
                    fullest = max(buckets, key=lambda k: len(buckets[k]))
                    yield batch_of(buckets[fullest])
                    buckets[fullest] = []
    for bucket in buckets.values():
        if bucket:
            yield batch_of(bucket)


class Postprocessor:
    """Runs ``postprocess`` per chip on one background thread, so the device keeps working.
    After each batch the runner calls :meth:`wait_for_room` (outside the device lock, as a
    postprocess may run MPS predictions itself), which waits until at most ``limit`` chips
    are waiting: a postprocess slower than the model holds prediction back instead of letting
    finished outputs pile up in memory."""

    def __init__(self, func: Callable[[Any], Any], executor: ThreadPoolExecutor, limit: int):
        self.func, self.executor, self.limit = func, executor, max(1, limit)
        self.pending: deque[Future] = deque()

    def submit(self, chip: Any) -> Future:
        """Queue a chip, without waiting (the caller may hold the device lock)."""
        future = self.executor.submit(self.func, chip)
        self.pending.append(future)
        return future

    def wait_for_room(self) -> None:
        # one worker runs them in order, so the oldest finishes first
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()  # raises its error here
        while len(self.pending) > self.limit:
            self.pending.popleft().result()


class GroupedSink:
    """Keeps each chip's finished outputs by input index (shapes may differ between
    batches)."""

    to_host = True

    def __init__(
        self,
        specs: Callable[[], dict[str, OutputSpec]],
        mask: bool,
        pbar: tqdm,
        post: Postprocessor | None,
        factors: Mapping[str, int] | None = None,
    ):
        self.specs = specs
        self.mask = mask
        self.pbar = pbar
        self.post = post
        self.factors = factors
        self.chips: dict[int, Any] = {}

    def __call__(self, prediction: Prediction) -> None:
        outputs = {
            name: x.numpy(force=True)
            for name, x in finish_batch(prediction, self.specs(), self.mask, self.factors).items()
        }
        for i, (index, _) in enumerate(prediction.items):
            chip = {name: x[i] for name, x in outputs.items()}
            if self.post is not None:
                self.chips[index] = self.post.submit(unwrap(chip))
            else:
                self.chips[index] = chip
        self.pbar.update(len(prediction.items))

    def after_batch(self) -> None:
        """Called by the runner between batches, outside the device lock."""
        if self.post is not None:
            self.post.wait_for_room()

    def result(self) -> Any:
        ordered = [self.chips[i] for i in sorted(self.chips)]
        if self.post is not None:
            return [future.result() for future in ordered]
        return [unwrap(chip) for chip in ordered]


def _describe(shapes: dict[str, tuple[int, ...]]) -> str:
    return str(shapes[SINGLE]) if set(shapes) == {SINGLE} else str(shapes)


def iter_array_batches(data: Any, batch_size: int) -> Iterator[RawBatch]:
    """Batches of a 4D array or tensor (or a dict of stacked arrays, one per input) as
    slices, without stacking."""
    inputs = as_inputs(data)
    count = {len(x) for x in inputs.values()}
    if len(count) != 1:
        raise ValueError(f"Every input needs the same number of chips, got {sorted(count)}.")
    _check_image({name: tuple(x.shape[1:]) for name, x in inputs.items()})
    total = count.pop()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield RawBatch(
            {name: x[start:stop] for name, x in inputs.items()},
            [None] * (stop - start),
            list(range(start, stop)),
        )


def output_mask(
    valid: torch.Tensor, shape: tuple[int, ...], factor: int | None = None
) -> torch.Tensor | None:
    """The (B, 1, h, w) valid mask for a dense output of ``shape`` (B, C, h, w): the input
    mask, or for a reduced-resolution output (h = ceil(H / f)) valid where any input pixel of
    the f x f block is valid. ``factor`` is f when the model's spec declares it; otherwise it
    is inferred from the sizes, and sizes more than one f fits (a 5 x 5 chip's 2 x 2 output
    is f = 3 or 4) are an error. None for image-level outputs."""
    if len(shape) != 4:
        return None
    height, width = valid.shape[-2:]
    if shape[-2:] == (height, width):
        return valid
    if shape[-2:] == (1, 1):  # every f from the chip size up gives one block: the whole chip
        return valid.flatten(2).any(dim=2)[..., None, None]
    factor = dense_factor((height, width), tuple(shape[-2:]), factor)
    pooled = F.max_pool2d(valid.float(), factor, stride=factor, ceil_mode=True)
    return pooled > 0


def dense_factor(grid: tuple[int, int], size: tuple[int, int], declared: int | None) -> int:
    """How many times coarser than the (H, W) ``grid`` a dense output of (h, w) ``size`` is:
    ``declared`` (checked), else 1 at the grid's size or the only f with h = ceil(H / f) and
    w = ceil(W / f). Raises ValueError when no f, or several, fit."""
    if tuple(size) == tuple(grid) and declared is None:
        return 1
    try:
        factor = _factor(tuple(grid), tuple(size), declared)
    except ValueError as error:
        verb = "Check" if declared is not None else "Declare"
        raise ValueError(
            f"{error} {verb} the output's scale in the model's tiling_spec() (or pass spec=)."
        ) from None
    if factor is None:
        raise ValueError(
            f"A dense output of size {tuple(size)} does not match the input size "
            f"{tuple(grid)}: dense outputs must be the input size, or "
            f"ceil(H / f) x ceil(W / f) for one integer f on both axes. Check the model's "
            f"output size."
        )
    return factor


def output_factors_of(
    outputs: Mapping[str, torch.Tensor], grid: tuple[int, int], declared: Mapping[str, int] | None
) -> dict[str, int | None]:
    """``{output: f}`` of a batch's outputs, as tiled mode reports them: how many times coarser
    than ``grid`` each dense output is (1 at full resolution), None for image-level ones."""
    return {
        name: None
        if x.ndim == 2
        else dense_factor(grid, tuple(x.shape[-2:]), _factor_of(declared, name))
        for name, x in outputs.items()
    }


def _factor_of(factors: Mapping[str, int] | None, name: str) -> int | None:
    """The reduction the spec declares for output ``name`` (a bare tensor's is the spec's only
    one), or None when it gives no scale (``factors`` holds only the scales a spec gives, so
    an explicit 1 is checked like any other)."""
    if not factors:
        return None
    if name == SINGLE and name not in factors and len(factors) == 1:
        factor = next(iter(factors.values()))
    else:
        factor = factors.get(name)
    return factor


def finish_batch(
    prediction: Prediction,
    specs: dict[str, OutputSpec],
    mask: bool,
    factors: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Final outputs of a batch, on the device: images without any valid pixel get no
    prediction (NaN, as all-nodata patches), then each output's mode, then nodata masking.
    ``"argmax"`` of a NaN the model returned where there is data is an error. ``factors``:
    ``{output: f}`` of the dense outputs the model's spec declares."""
    out = {}
    empty = prediction.empty
    for name, x in prediction.outputs.items():
        factor = _factor_of(factors, name)
        if x.ndim == 4 and factor is not None:
            # a declared scale is checked whether or not masking needs it, as in tiled mode
            dense_factor(tuple(prediction.valid.shape[-2:]), tuple(x.shape[-2:]), factor)
        chips = empty.view(-1, *[1] * (x.ndim - 1))
        if specs[name].is_class:
            data = output_mask(prediction.valid, tuple(x.shape), factor)
            check_argmax(x, ~chips if data is None else data, name)
        x = torch.where(chips, torch.nan, x)
        x = specs[name].finalize(x, class_dim=1)
        valid = output_mask(prediction.valid, tuple(x.shape), factor) if mask else None
        out[name] = mask_nodata(x, valid) if valid is not None else x
    return out


class BatchSink:
    """Finishes each output and copies it to the host, into preallocated arrays when the
    number of images is known. With ``postprocess`` no outputs are kept: each chip goes to it
    and only its results are."""

    to_host = True

    def __init__(
        self,
        specs: Callable[[], dict[str, OutputSpec]],
        total: int | None,
        mask: bool,
        pbar: tqdm,
        factors: Mapping[str, int] | None = None,
    ):
        self.specs = specs
        self.total = total
        self.mask = mask
        self.pbar = pbar
        self.factors = factors
        self.arrays: dict[str, np.ndarray] = {}
        self.parts: dict[str, list[np.ndarray]] = {}
        self.written = 0
        self.post: Postprocessor | None = None
        self.post_results: list[Future] = []

    def __call__(self, prediction: Prediction) -> None:
        n = len(prediction.items)
        stored = {}
        for name, x in finish_batch(prediction, self.specs(), self.mask, self.factors).items():
            if self.post is not None:
                # this batch only: postprocess takes each chip, and its results are returned
                stored[name] = x.numpy(force=True)
                continue
            if self.total is None:
                self.parts.setdefault(name, []).append(x.numpy(force=True))
                stored[name] = self.parts[name][-1]
                continue
            if name not in self.arrays:
                dtype = torch.empty(0, dtype=x.dtype).numpy().dtype
                self.arrays[name] = np.empty((self.total, *x.shape[1:]), dtype=dtype)
            # copy straight into the output, without an intermediate array
            stored[name] = self.arrays[name][self.written : self.written + n]
            torch.from_numpy(stored[name]).copy_(x)
        if self.post is not None:
            for i in range(n):
                chip = unwrap({name: a[i] for name, a in stored.items()})
                self.post_results.append(self.post.submit(chip))
        self.written += n
        self.pbar.update(n)

    def after_batch(self) -> None:
        """Called by the runner between batches, outside the device lock."""
        if self.post is not None:
            self.post.wait_for_room()

    def result(self) -> dict[str, np.ndarray]:
        if self.total is None:
            return {name: np.concatenate(parts) for name, parts in self.parts.items()}
        return self.arrays


def unwrap(outputs: dict[str, Any]) -> Any:
    """A single-output model's result as the array itself, else the dict."""
    return outputs[SINGLE] if set(outputs) == {SINGLE} else outputs


def one_raster_spec(normalize: Any, rasters: list[str]) -> Any:
    """A spec that isn't per input, as ``{raster: spec}`` when there is one raster to apply it
    to (vectors aren't normalised, so one raster beside a vector is unambiguous), and as None
    when it has no effective steps."""
    if rasters == [SINGLE]:
        return single_input_spec(normalize)
    per_input = isinstance(normalize, Mapping) and "type" not in normalize
    if normalize is None or normalize is False or per_input:
        return normalize
    if all(
        isinstance(step, Mapping) and step.get("type") == "none" for step in steps_of(normalize)
    ):
        return None  # no steps can't be ambiguous, whatever the number of rasters
    return {rasters[0]: normalize} if len(rasters) == 1 else normalize


def single_input_spec(normalize: Any) -> Any:
    """For a model given one unnamed input: a per-input ``{name: spec}`` with one entry (a
    stored normalisation names the model's input) as that spec."""
    if isinstance(normalize, Mapping) and "type" not in normalize and len(normalize) == 1:
        (steps,) = normalize.values()
        if steps is None or steps is False or callable(steps) or isinstance(steps, Mapping):
            return steps
        if isinstance(steps, Sequence) and not isinstance(steps, str):
            return steps
    return normalize


def build_normalizers(
    normalize: NormalizeSpec | dict[str, NormalizeSpec], names: list[str], batch_mode: bool
) -> dict[str, Any]:
    """{input_name: normalizer}; a dict spec is per input, anything else is for the single
    input. None or False normalises nothing."""
    if names == [SINGLE]:
        normalize = single_input_spec(normalize)
    if normalize is None or normalize is False:
        return {}
    # for a single-input model a dict without "type" is a malformed step, not per-input specs
    if isinstance(normalize, Mapping) and "type" not in normalize and names != [SINGLE]:
        unknown = sorted(set(normalize) - set(names))
        if unknown:
            raise ValueError(f"normalize given for unknown inputs {unknown}; inputs: {names}.")
        return {
            name: build_normalizer(spec, batch_mode=batch_mode) for name, spec in normalize.items()
        }
    if names != [SINGLE]:
        raise ValueError("This model has several inputs; pass normalize as {input_name: spec}.")
    return {SINGLE: build_normalizer(normalize, batch_mode=batch_mode)}


def predict_batch(
    model: Any,
    chips: Any,
    *,
    batch_size: int = 8,
    load_func: Callable | None = None,
    group_by_shape: bool = False,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = "float32",
    compile: bool = False,
    normalize: Any = None,
    activation: Any = None,
    output: Any = "auto",
    nodata: float | None = None,
    mask_nodata: bool = True,
    postprocess: Callable[[Any], Any] | None = None,
    spec: dict[str, Any] | None = None,
    progress: bool = True,
) -> Any:
    """Predict many same-size chips, each as a whole (no tiling or blending).

    Args:
        model: An ``nn.Module`` or callable taking a (B, C, H, W) tensor and returning a tensor
            or a dict of tensors, or a list of them (an ensemble: raw outputs are averaged).
            Modules are moved to ``device`` and ``dtype`` in place and set to eval mode; on
            MPS their ``Conv2d`` layers also become a subclass that makes inputs contiguous
            (it stays, and pickles as a plain ``Conv2d``). A callable that isn't a module may
            declare ``device`` (the default device), ``batch_size`` (replaces
            ``batch_size``; short batches are padded by repeating their last item) and
            ``fixed_size`` (the chip size it takes).
        chips: A 4D (N, C, H, W) array or tensor (sliced without copying), any iterable of
            (C, H, W) arrays or tensors, or, with ``load_func``, an iterable of items to load.
            For a multi-input model: a dict of stacked arrays, ``{input_name: (N, ...)}``, or
            an iterable of ``{input_name: array}`` dicts, one per chip.
        batch_size: Chips per batch.
        load_func: ``load_func(item) -> image`` or ``(image, meta)``, called on background
            threads; it must return CPU data (numpy arrays, CPU tensors, or lazy readers, which
            are read whole).
        group_by_shape: Accept chips of different shapes: they are bucketed by shape and
            each bucket is batched on its own (at most ``4 * batch_size`` chips wait in
            partial buckets; past that the fullest goes as a smaller batch). Results come
            back in input order as a list with one array (or dict) per chip, whatever the
            shapes turn out to be.
        device: Inference device; default CUDA, then MPS, then the CPU.
        dtype: Inference dtype: ``"float32"`` (the default), ``"float64"`` (not on MPS), or
            ``"float16"`` or
            ``"bfloat16"``,
            which are faster and use less GPU memory on hardware that supports them but round
            the model's weights and outputs: outputs in large units lose the most (values around
            1500 can be off by units). The weights of an ``nn.Module`` stay rounded after the call.
            Inputs are normalised in float32; float outputs are float32 whatever the dtype, and
            ``"argmax"`` classes are uint8 (int64 past 256 classes).
        compile: ``torch.compile`` the models, with static shapes (warmed up on the first
            batch's shape; shorter batches are padded to it, which assumes chips don't
            affect each other's outputs, as batching does anyway).
        normalize: Normalisation spec (see :mod:`manypred.normalize`); None uses the model's
            stored normalisation (``tiling_spec()``) if it has one, else no steps; False always
            means no steps. Either way values arrive as float32, with missing ones (``nodata``,
            NaN, infinity) as 0. In batch mode ``scope="tile"`` statistics are per chip.
        activation: ``"softmax"``, ``"sigmoid"``, a callable, ``"none"`` or None, or
            ``{output: ...}``. None uses the activation the model declares, if any; ``"none"``
            means no activation.
        output: ``"auto"`` (the default: ``"probs"`` for outputs with an activation, given
            or declared in the model's ``tiling_spec()``, else ``"logits"``), ``"logits"``,
            ``"probs"``, ``"argmax"`` or a callable
            on the (B, C, ...) float32 predictions, or ``{output: ...}``. ``"argmax"`` raises a
            RuntimeError if the model returned NaN at a pixel with data. See
            :mod:`manypred.outputs`.
        nodata: Input value marking missing data (NaN allowed), or None (the default: no
            value does; NaN and infinity always count as missing). A pixel is nodata when every
            band is.
            The model sees missing values as 0.
        mask_nodata: Mark nodata pixels in dense outputs: NaN in float outputs, 0 in class
            outputs. Chips that are entirely nodata are NaN in float outputs and 0 in class
            outputs either way. Class 0 is also a real class, so ``"argmax"`` outputs can't
            tell nodata from it: keep the input's nodata mask for that.
        spec: A tiling spec (:mod:`manypred.spec`); default ``model.tiling_spec()``. Batch
            mode uses its output names and activations (as defaults for ``activation`` and
            ``output``) and its stored normalisation (when ``normalize`` is None), not its
            tiling settings.
        postprocess: ``f(chip_result) -> anything``, run per chip on a background thread
            (``chip_result`` is that chip's array, or dict of arrays). Only its results are
            kept, so it can save chips without holding every output in memory.
        progress: Show a progress bar.

    Returns:
        For a single-output model an array, (N, C, h, w) for dense outputs (at their own
        resolution), (N, 1, h, w) uint8 for ``"argmax"`` (int64 past 256 classes), (N, C)
        for image-level outputs.
        For a dict-output model, a dict of such arrays. With ``group_by_shape`` a list with one
        array (or dict) per chip; with ``postprocess``, a list of its results, one per chip.
    """
    batch_size = whole("batch_size", batch_size, minimum=1)
    setup = set_up_run(
        model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        default_batch_size=8,
        spec=spec,
        normalize=normalize,
        activation=activation,
        output=output,
        nodata=nodata,
        compile=compile,
    )
    runner, batch_size = setup.runner, setup.batch_size

    try:
        total: int | None = len(chips)
    except TypeError:
        total = None

    # stacked chips, sliced without copying; with load_func an array holds items to load
    stacked = load_func is None and (
        isinstance(chips, (np.ndarray, torch.Tensor))
        or (isinstance(chips, Mapping) and all(hasattr(x, "shape") for x in chips.values()))
    )
    if (
        isinstance(chips, Mapping)
        and stacked
        and (
            any(len(x.shape) < 2 for x in chips.values())
            or not any(len(x.shape) >= 4 for x in chips.values())
        )
    ):
        raise ValueError(
            "chips as a dict must hold one stacked (N, ...) array or tensor per input "
            "((N, C, H, W) rasters, (N, D) vectors); this looks like a single chip. Pass a "
            "list of {input_name: array} dicts, one per chip."
        )
    if isinstance(chips, Mapping) and not stacked and load_func is None:
        raise ValueError(
            "chips as a dict must hold one stacked (N, ...) array or tensor per input; for "
            "per-chip dicts pass a list of {input_name: array} instead."
        )
    if stacked:
        if not isinstance(chips, Mapping) and chips.ndim != 4:
            raise ValueError(
                f"chips as an array must be (N, C, H, W), got shape {tuple(chips.shape)}. Pass "
                f"(N, T, C, H, W) time series in a dict, {{name: series}}."
                if chips.ndim == 5
                else f"chips as an array must be (N, C, H, W), got shape {tuple(chips.shape)}."
            )
        batches = iter_array_batches(chips, batch_size)
        first = next(batches, None)
        if first is None:
            raise ValueError("chips is empty; pass at least one chip.")
        names = list(first.inputs)
        if isinstance(chips, Mapping):
            total = len(next(iter(chips.values())))
    else:
        iterate = iter_grouped_batches if group_by_shape else iter_item_batches
        batches = iterate(chips, load_func, batch_size, order=input_order(setup.tiling))
        first = next(batches, None)
        if first is None:
            raise ValueError("chips is empty; pass at least one chip.")
        names = list(first.inputs)

    runner.normalizers = build_normalizers(
        one_raster_spec(setup.normalize, raster_names(first)), names, batch_mode=True
    )
    runner.primary = primary_input(first, input_order(setup.tiling))

    def all_batches() -> Iterator[RawBatch]:
        yield first
        yield from batches

    pbar = tqdm(
        total=total,
        desc=f"Predicting on {runner.device.type} {str(runner.dtype).split('.')[-1]}",
        disable=not progress,
    )
    executor = ThreadPoolExecutor(max_workers=1) if postprocess is not None else None
    post = Postprocessor(postprocess, executor, 2 * batch_size) if executor is not None else None
    specs = lambda: runner.specs or {}  # noqa: E731
    factors = named_output_factors(setup.tiling)
    grouped = group_by_shape and not stacked
    sink: BatchSink | GroupedSink
    if grouped:
        sink = GroupedSink(specs, mask_nodata, pbar, post, factors)
    else:
        sink = BatchSink(specs, total, mask_nodata, pbar, factors)
        sink.post = post
    ok = False
    try:
        runner.run(all_batches(), sink)
        ok = True
    finally:
        pbar.close()
        if executor is not None:
            executor.shutdown(wait=ok, cancel_futures=not ok)
    if isinstance(sink, GroupedSink):
        return sink.result()
    if postprocess is not None:
        return [future.result() for future in sink.post_results]
    result = sink.result()
    if group_by_shape:  # a stacked input: still one entry per chip
        return [unwrap({name: x[i] for name, x in result.items()}) for i in range(total)]
    return unwrap(result)
