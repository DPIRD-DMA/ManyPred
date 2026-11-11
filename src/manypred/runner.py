"""The core runner: batches in, activated predictions out, on the device.

iterate -> collate (background thread; pinned host buffers, small dtypes kept) -> H2D on a copy
stream -> float32 on the device -> valid mask -> normalise (float32) -> cast to the inference
dtype -> model(s), ensemble mean -> float32 -> activation -> sink.

Sinks that copy predictions to the host get them one batch late on CUDA, inside a separate copy
stream, so the GPU keeps working while the host stores results. Batch mode and tiled mode are
both "a stream of same-size batches through the runner" with different sinks.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from queue import Full, Queue
from threading import Event, RLock, Thread
from typing import Any, Protocol, cast

import torch

from .collate import (
    SINGLE,
    chip_dtypes,
    chip_nodata,
    collate,
    tensor_devices,
    to_float32,
    touches_mps,
)
from .devices import (
    device_lock,
    inference_dtype,
    patch_for_mps,
    resolve_device,
    same_device,
    warn_caller,
)
from .normalize import Normalizer, band_valid_mask, build_normalizer
from .outputs import OutputSpec, make_specs
from .spec import (
    TilingSpec,
    get_tiling_spec,
    output_activations,
    output_names,
    regression_outputs,
    vector_inputs,
)


@dataclass
class RawBatch:
    """Images of one batch, per input: a list of arrays/tensors or a stacked array/tensor."""

    inputs: dict[str, Any]
    metas: list[Any]
    items: list[Any]


@dataclass
class Batch:
    """A collated batch. ``is_uint16`` is None while the batch still needs collating (it
    touches MPS, so it is collated under the device lock)."""

    inputs: dict[str, Any]
    is_uint16: dict[str, bool] | None
    metas: list[Any]
    items: list[Any]
    ready: Any = None  # CUDA event: the copy to the device is done
    # {input: each chip's dtype} of inputs whose chips mixed dtypes (see collate.chip_dtypes)
    dtypes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """One batch of activated predictions, on the device.

    ``outputs``: {name: float32 (B, C, ...)}; ``valid``: (B, 1, H, W) bool, pixels of the
    primary input where some band is not nodata; ``empty``: (B,) bool, images with no valid
    pixel at all (computed when first used: tiled mode never needs it)."""

    outputs: dict[str, torch.Tensor]
    valid: torch.Tensor
    metas: list[Any]
    items: list[Any]

    @property
    def empty(self) -> torch.Tensor:
        return ~self.valid.flatten(start_dim=1).any(dim=1)


class Sink(Protocol):
    """Receives predictions. ``to_host``: whether it copies them to the host (then, on CUDA,
    it is called one batch late inside a copy stream). A sink may also define
    ``after_batch()``, called after each batch outside the device lock, for waits that must
    not hold it."""

    to_host: bool

    def __call__(self, prediction: Prediction) -> None: ...


_SENTINEL = object()
_ISFINITE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
# Threads may share a model. Casting one replaces its tensors, so only one thread at a time may.
_CAST_LOCK = RLock()
# Modules, parameters and buffers some run is using: {id: [device, dtype, number of runs, the
# object]}. A run elsewhere may not recast them. Keyed by identity, so a submodule or tensor
# shared between two models is caught whichever wrapper casts it; an entry holds its object, so
# the id stays its own until the last run using it ends and removes the entry
_IN_USE: dict[int, list[Any]] = {}
# How long closing a prefetch waits for its loader thread after an error before leaving it
_STOP_TIMEOUT = 5.0
_FILL_MISSING = build_normalizer(None)  # for inputs with no normalisation steps


def prefetch(batches: Iterator[RawBatch], device: torch.device, depth: int = 2) -> Iterator[Batch]:
    """Collate batches in a background thread so the next batch is ready while the current one
    runs. On CUDA the thread also copies each batch to the device on its own stream. Batches
    whose collation touches MPS are passed on uncollated. Errors are re-raised in the caller,
    and closing this generator stops the thread. Closed early (an error in the caller, or
    Ctrl-C), it waits at most a few seconds for a loader blocked in a read: the daemon thread
    then stops on its own once the read returns."""
    queue: Queue = Queue(maxsize=depth)
    stop = Event()
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def put(item: Any) -> bool:
        # time out regularly so the producer can't block forever on a full queue
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def producer() -> None:
        try:
            for raw in batches:
                if any(touches_mps(images) for images in raw.inputs.values()):
                    if not put(Batch(raw.inputs, None, raw.metas, raw.items)):
                        return
                    continue
                inputs, is_uint16, dtypes = {}, {}, {}
                for name, images in raw.inputs.items():
                    if (mixed := chip_dtypes(images)) is not None:
                        dtypes[name] = mixed
                    inputs[name], is_uint16[name] = collate(images, device)
                ready = None
                if copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        for name, tensor in inputs.items():
                            if tensor.device.type == "cpu":
                                inputs[name] = tensor.to(device=device, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(copy_stream)
                if not put(Batch(inputs, is_uint16, raw.metas, raw.items, ready, dtypes)):
                    return
            put(_SENTINEL)
        except BaseException as e:
            put(e)
        finally:
            close = getattr(batches, "close", None)
            if close is not None:
                close()

    thread = Thread(target=producer, daemon=True)
    thread.start()
    finished = False
    try:
        while True:
            result = queue.get()
            if result is _SENTINEL:
                finished = True
                return
            if isinstance(result, BaseException):
                raise result
            if result.ready is not None:
                # wait for the copy, and tell the allocator the tensors are used on this stream
                compute = torch.cuda.current_stream(device)
                compute.wait_event(result.ready)
                for tensor in result.inputs.values():
                    tensor.record_stream(compute)
            yield result
    finally:
        stop.set()
        # after the sentinel the thread is only closing ``batches``
        thread.join(None if finished else _STOP_TIMEOUT)


def whole(name: str, value: Any, minimum: int | None = None) -> int:
    """``value`` as an int, or a TypeError naming the argument (floats and bools aren't), and a
    ValueError if it is below ``minimum``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an integer, got {value!r} ({type(value).__name__})."
        ) from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}.")
    return number


def as_model_list(model: Any) -> list[Any]:
    """A model, or a list/tuple of them (an ensemble), as a non-empty list."""
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models:
        raise ValueError("Pass at least one model.")
    for m in models:
        if not callable(m):
            raise TypeError(f"Models must be torch.nn.Module or callables, got {type(m)}.")
    return models


def _program_attr(model: Any, name: str) -> Any:
    """What a program declares about itself. A program is a callable that isn't an ``nn.Module``
    and has fixed requirements, such as ``manyseg.load``'s ``QuantizedProgram``. Modules are left
    out, as they may use these names for something else."""
    return None if isinstance(model, torch.nn.Module) else getattr(model, name, None)


def _one_value(
    models: Sequence[Any],
    name: str,
    convert: Callable[[Any], Any],
    same: Callable[[Any, Any], bool] = operator.eq,
) -> Any:
    """The value the programs among ``models`` agree on for ``name`` (by ``same``), or None if
    none sets it."""
    values = [convert(v) for v in (_program_attr(m, name) for m in models) if v is not None]
    if any(not same(v, values[0]) for v in values):
        raise ValueError(f"The models in this ensemble need different {name}: {values}.")
    return values[0] if values else None


def program_device(models: Sequence[Any], device: str | torch.device | None) -> Any:
    """``device`` if given, else the device the programs run on (None if there are none)."""
    if device is not None:
        return device
    # "mps" and "mps:0" (or "cuda" and the current "cuda:N") are one device
    return _one_value(models, "device", torch.device, same_device)


def program_batch_size(models: Sequence[Any], batch_size: int, default: int | None) -> int:
    """The batch size a program needs, or ``batch_size``. Short batches are padded to it by
    repeating their last item (:class:`_FullBatches`), which assumes the program treats each
    item on its own. Warns when the caller asked for another size (``default`` is the
    function's default, which isn't asking; None if ``batch_size`` was certainly asked for)."""
    size = _one_value(models, "batch_size", int)
    if size is None:
        return batch_size
    if batch_size not in (size, default):
        warn_caller(f"This model was exported for batch size {size}; using it, not {batch_size}.")
    return size


def program_patch_size(
    models: Sequence[Any], patch_size: int, overlap: int, adapt_patch_size: bool
) -> tuple[int, int]:
    """Tiling a program with a fixed input size: the patch is that size, and only that size.
    Call this after :func:`~manypred.spec.fit_patch`, as a spec can't change what a program
    takes."""
    fixed = _one_value(models, "fixed_size", tuple)
    if fixed is None:
        return patch_size, overlap
    height, width = fixed
    if height != width:
        raise ValueError(
            f"This model was exported at a fixed {height} x {width}, and tiling uses square "
            f"patches. Export it with dynamic=True or at a square size to tile with it."
        )
    if adapt_patch_size:
        raise ValueError(
            f"adapt_patch_size can't shrink patches for a model exported at a fixed "
            f"{height} x {width}. Pass adapt_patch_size=False."
        )
    if patch_size == height:
        return patch_size, overlap
    overlap = min(overlap, height // 2)
    warn_caller(
        f"This model was exported at a fixed {height} x {width}; using patch_size={height} and "
        f"overlap={overlap}, not patch_size={patch_size}."
    )
    return height, overlap


def ensemble(models: Sequence[Any], x: Any) -> dict[str, torch.Tensor]:
    """Run each model and average their raw outputs (as OmniCloudMask does). Returns
    {name: tensor}; a model returning a tensor gives {SINGLE: tensor}."""
    outputs = []
    for model in models:
        out = model(x)
        if isinstance(out, torch.Tensor):
            out = {SINGLE: out}
        elif not isinstance(out, Mapping) or not out:
            raise TypeError(
                f"Models must return a tensor or a non-empty dict of tensors, got "
                f"{type(out).__name__ if not isinstance(out, Mapping) else 'an empty dict'}."
            )
        for name, value in out.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Model output {name!r} is a {type(value).__name__}; outputs must be tensors."
                )
        outputs.append(out)
    if len(outputs) == 1:
        return dict(outputs[0])
    names = list(outputs[0])
    if any(set(out) != set(names) for out in outputs[1:]):
        raise ValueError(
            f"Models in an ensemble must return the same outputs, got "
            f"{[sorted(out) for out in outputs]}."
        )
    return {name: torch.mean(torch.stack([out[name] for out in outputs]), dim=0) for name in names}


def is_dates(name: str, names: Iterable[str]) -> bool:
    """Whether ``name`` is the ``<input>_dates`` of an input in ``names``."""
    return name.endswith("_dates") and name.removesuffix("_dates") in names


def cast_input(
    name: str,
    t: torch.Tensor,
    names: Iterable[str],
    dtype: torch.dtype,
    exact: Collection[str] = (),
) -> torch.Tensor:
    """``t`` in the inference dtype. Float dates stay as they are: 16-bit floats can't hold a
    day count (bfloat16 rounds 18000 and 18050 to the same value), and the model sees no error,
    only dates that have stopped mattering. So do the ``exact`` inputs, the vectors a tiling
    spec declares (see :func:`~manypred.spec.vector_inputs`): bfloat16 would round a category
    id of 257 to 256."""
    keep = is_dates(name, names) or name in exact
    return t.to(dtype) if t.is_floating_point() and not keep else t


def _vector_example(
    name: str,
    t: torch.Tensor,
    names: Iterable[str],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    exact: Collection[str] = (),
) -> torch.Tensor:
    return cast_input(name, t[:1].to(device).expand(size, -1), names, dtype, exact)


class _FullBatches:
    """A compiled model called with every batch padded to ``size`` items (repeating the last),
    its outputs cut back: one compiled shape per patch size instead of one per batch size."""

    def __init__(self, compiled: Any, size: int):
        self.compiled, self.size = compiled, size

    def __call__(self, x: Any) -> Any:
        tensors = x if isinstance(x, dict) else {SINGLE: x}
        n = next(iter(tensors.values())).shape[0]
        if n >= self.size:
            return self.compiled(x)

        def pad(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([t, t[-1:].expand(self.size - n, *t.shape[1:])])

        padded = {k: pad(t) for k, t in tensors.items()}
        out = self.compiled(padded if isinstance(x, dict) else padded[SINGLE])
        if isinstance(out, torch.Tensor):
            return out[:n]
        if isinstance(out, dict):
            return {k: v[:n] if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        if isinstance(out, (list, tuple)):
            return type(out)(v[:n] if isinstance(v, torch.Tensor) else v for v in out)
        return out


def _would_round(model: torch.nn.Module, dtype: torch.dtype) -> bool:
    """Whether casting to ``dtype`` loses precision in any floating parameter or buffer."""
    bits = torch.finfo(dtype).bits
    return any(
        t.is_floating_point() and t.dtype != dtype and torch.finfo(t.dtype).bits >= bits
        for t in (*model.parameters(), *model.buffers())
    )


def _canonical(device: torch.device) -> torch.device:
    """``device`` as tensors report it: "cuda" as "cuda:0"."""
    return torch.empty(0, device=device).device


def _is_cast(model: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> bool:
    """Whether every parameter and buffer is already on ``device``, floating ones in ``dtype``,
    so that casting would change nothing."""
    tensors = [*model.parameters(), *model.buffers()]
    device = _canonical(device)
    return bool(tensors) and all(
        t.device == device and (t.dtype == dtype or not t.is_floating_point()) for t in tensors
    )


def _cast_targets(modules: Sequence[torch.nn.Module]) -> list[Any]:
    """Everything casting ``modules`` changes: each module and submodule, parameter and buffer,
    once each."""
    targets: dict[int, Any] = {}
    for model in modules:
        for item in (*model.modules(), *model.parameters(), *model.buffers()):
            targets.setdefault(id(item), item)
    return list(targets.values())


@contextmanager
def _claimed(modules: Sequence[torch.nn.Module], device: torch.device, dtype: torch.dtype):
    """Mark ``modules`` (with their submodules, parameters and buffers) as in use on
    ``device`` in ``dtype`` while the block runs. Raises if another run (in another thread) is
    using any of them on another device or in another dtype: casting it here would change its
    weights under that run."""
    device = _canonical(device)
    targets = _cast_targets(modules)
    with _CAST_LOCK:
        for item in targets:
            used = _IN_USE.get(id(item))
            if used is not None and (used[0], used[1]) != (device, dtype):
                raise RuntimeError(
                    f"This model is being run on {used[0]} in "
                    f"{str(used[1]).removeprefix('torch.')} by another call, and this call needs "
                    f"{device} in {str(dtype).removeprefix('torch.')}; running it would recast "
                    f"its weights under the other call. Give each call its own copy "
                    f"(copy.deepcopy(model)), or use the same device and dtype."
                )
        for item in targets:
            _IN_USE.setdefault(id(item), [device, dtype, 0, item])[2] += 1
    try:
        yield
    finally:
        with _CAST_LOCK:
            for item in targets:
                used = _IN_USE[id(item)]
                used[2] -= 1
                if not used[2]:
                    del _IN_USE[id(item)]


def warn_not_classes(specs: Mapping[str, OutputSpec]) -> None:
    """Warn about regression outputs that a global ``output="argmax"`` leaves as values."""
    for name, spec in specs.items():
        if spec.not_classes:
            label = repr(name) if name else "of this model"
            warn_caller(
                f"output='argmax' does not apply to regression output {label}: its channels "
                f"aren't classes, so it keeps its values. Set output per output to silence this."
            )


def check_outputs(outputs: dict[str, torch.Tensor], batch_size: int) -> None:
    """Outputs must be (B, C) image-level or (B, C, h, w) dense tensors for the whole batch."""
    for name, x in outputs.items():
        label = "The model output" if name == SINGLE else f"Model output {name!r}"
        if x.ndim not in (2, 4) or x.shape[0] != batch_size:
            raise ValueError(
                f"{label} has shape {tuple(x.shape)} for a batch of {batch_size}; outputs "
                f"must be (B, C) for image-level heads or (B, C, h, w) for dense heads."
            )


@dataclass
class Runner:
    """Runs models over a stream of batches. See the module docstring.

    Args:
        models: Models (``nn.Module`` or callables); several are averaged.
        device: Inference device.
        dtype: Inference dtype; inputs are normalised in float32 and cast to it.
        normalizers: {input_name: f(x, valid) -> x}; inputs without one get float32 values.
        nodata: Nodata value of the inputs (NaN allowed), or None; NaN is always missing.
        primary: Input whose pixels define the valid mask.
        make_specs: Builds the {output_name: OutputSpec} once the output names are known.
        compile: ``torch.compile`` the models with static shapes, warmed up on the first
            batch; shorter batches are padded to its size (repeating the last item) and the
            outputs cut back.

    ``fixed_size``: the (height, width) input size the programs among the models declare, or
    None.
    """

    models: list[Any]
    device: torch.device
    dtype: torch.dtype
    normalizers: dict[str, Normalizer]
    nodata: float | None
    primary: str
    make_specs: Callable[[list[str]], dict[str, OutputSpec]]
    compile: bool = False
    # vectors the tiling spec declares: passed in float32 whatever the dtype (see cast_input)
    exact_vectors: frozenset[str] = frozenset()
    specs: dict[str, OutputSpec] | None = field(default=None, init=False)
    fixed_size: tuple[int, int] | None = field(default=None, init=False)
    _prepared: bool = field(default=False, init=False)
    _modules: list[torch.nn.Module] = field(default_factory=list, init=False)
    # (items with bad outputs, items with data) of this run, kept on the device: reading them for
    # every batch would make the host wait for the GPU
    _tally: tuple[torch.Tensor, torch.Tensor] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # read before preparing, which wraps programs
        self.fixed_size = _one_value(self.models, "fixed_size", tuple)
        self._modules = [m for m in self.models if isinstance(m, torch.nn.Module)]

    def claim(self) -> Any:
        """Context manager marking this runner's modules as in use (see :func:`_claimed`):
        hold it from :meth:`prepare` until the last prediction."""
        return _claimed(self._modules, self.device, self.dtype)

    def prepare(self, batch: Batch, compile_size: int | None = None) -> None:
        """Move the models to the device and dtype (in place, as ``nn.Module.to`` does), set
        eval mode, patch them for MPS and compile them, once. ``compile_size``: the batch
        size to compile for, when ``batch`` is shorter than the batches that follow."""
        if self._prepared:
            return
        with _CAST_LOCK:
            prepared = []
            for model in self.models:
                if isinstance(model, torch.nn.Module):
                    # a model already cast is left alone: another thread may be running it
                    if not _is_cast(model, self.device, self.dtype):
                        if model.training and _would_round(model, self.dtype):
                            name = str(self.dtype).removeprefix("torch.")
                            warn_caller(
                                f"This model is in training mode, and dtype='{name}' rounds its "
                                f"weights in place, for good. Pass copy.deepcopy(model) to keep "
                                f"training it."
                            )
                        model = model.to(device=self.device, dtype=self.dtype)
                    model = model.eval()
                prepared.append(model)
            if self.device.type == "mps":
                patch_for_mps(prepared)
        if self.compile:
            prepared = [self._compile(model, batch, compile_size) for model in prepared]
        else:
            prepared = [self._full_batches(model) for model in prepared]
        self.models = prepared
        self._prepared = True

    def _full_batches(self, model: Any) -> Any:
        """A program with a fixed batch size takes every batch padded to it."""
        size = _program_attr(model, "batch_size")
        return model if size is None else _FullBatches(model, size)

    def _compile(self, model: Any, batch: Batch, size: int | None = None) -> Any:
        try:
            # static shapes: symbolic sizes have given wrong outputs for nearest upsampling on
            # CUDA. Short batches are padded to the full size so the batch size never triggers
            # a recompile
            compiled = torch.compile(model, fullgraph=False, dynamic=False)
            size = _program_attr(model, "batch_size") or size or len(batch.items)
            with torch.no_grad():
                # warm up on zeros of the first batch's shape
                example = {
                    # vectors (masks, dates, metadata) as in the batch, rasters as zeros
                    name: _vector_example(
                        name, t, batch.inputs, size, self.device, self.dtype, self.exact_vectors
                    )
                    if t.ndim == 2
                    else torch.zeros((size, *t.shape[1:]), device=self.device, dtype=self.dtype)
                    for name, t in batch.inputs.items()
                }
                compiled(example[SINGLE] if SINGLE in example else example)
            return _FullBatches(compiled, size)
        except Exception as e:
            warn_caller(f"torch.compile failed ({e}); using the model without compiling.")
            return self._full_batches(model)

    def _nodata_of(self, batch: Batch, name: str, x: torch.Tensor) -> Any:
        """``nodata``, or for an input whose chips mixed dtypes, a (B, 1, ...) tensor of it as
        each chip's dtype holds it, so each chip matches it as it would batched alone."""
        nodata, dtypes = self.nodata, batch.dtypes.get(name)
        if dtypes is None or nodata is None or not x.is_floating_point():
            return nodata
        if isinstance(nodata, float) and math.isnan(nodata):
            return nodata
        values = torch.tensor(chip_nodata(dtypes, nodata), dtype=torch.float64)
        return values.to(device=x.device, dtype=x.dtype).view(-1, *[1] * (x.ndim - 1))

    def predict(self, batch: Batch) -> Prediction:
        """Run one collated batch (call under the device lock)."""
        assert batch.is_uint16 is not None
        # Non-blocking copies are only safe from host memory to CUDA
        non_blocking = self.device.type == "cuda"
        x = {}
        integer = {}  # inputs that came as integers: no NaN or inf to check for
        masks = {}  # {input: band-level valid mask}, from the values as they came
        for name, tensor in batch.inputs.items():
            moved = tensor.to(device=self.device, non_blocking=non_blocking and tensor.is_pinned())
            integer[name] = not moved.is_floating_point()
            if moved.ndim == 2 and integer[name]:
                x[name] = moved  # vectors such as masks and dates keep their dtype
                continue
            x[name] = to_float32(moved, batch.is_uint16[name])
            needs_mask = (
                name == self.primary
                or name in self.normalizers
                or not integer[name]
                or self.nodata is not None
            )
            if moved.ndim > 2 and needs_mask:
                # nodata is compared before the float32 cast, which can merge values (int32
                # 16777217 and 16777216); uint16 travels as int16, and float32 holds it exactly
                native = x[name] if batch.is_uint16[name] else moved
                masks[name] = band_valid_mask(
                    native, self._nodata_of(batch, name, native), not integer[name]
                )
            del moved

        band_valid = masks[self.primary]
        if band_valid.ndim == 5:  # a time series: nodata when every frame and band is
            band_valid = band_valid.flatten(1, 2)
        # a pixel is nodata only if every band is nodata
        valid = band_valid.any(dim=1, keepdim=True)

        model_in = {}
        for name, tensor in x.items():
            normalize = self.normalizers.get(name)
            if tensor.ndim == 2:  # vectors pass through unnormalised
                if normalize is not None:
                    raise ValueError(
                        f"Input {name!r} is a vector; vectors are passed to the model as they "
                        f"are, so remove it from normalize."
                    )
                model_in[name] = tensor
                continue
            nan = not integer[name]
            if normalize is None and (nan or self.nodata is not None):
                normalize = _FILL_MISSING
            if normalize is not None:
                if tensor.ndim == 5:  # time series: normalise each frame
                    b, t = tensor.shape[:2]
                    frames = normalize(tensor.flatten(0, 1), masks[name].flatten(0, 1))
                    tensor = frames.unflatten(0, (b, t))
                else:
                    tensor = normalize(tensor, masks[name])
            model_in[name] = tensor
        del x, masks, band_valid

        raw = self._call(model_in)
        del model_in
        check_outputs(raw, len(batch.items))
        self._count_bad(raw, valid)
        if self.specs is None:
            specs = self.make_specs(list(raw))
            for name, tensor in raw.items():
                specs[name].check_shape(tensor.ndim)
            warn_not_classes(specs)
            self.specs = specs
        elif set(raw) != set(self.specs):
            raise ValueError(f"Model outputs changed between batches: {list(raw)}.")
        outputs = {name: self.specs[name].activate(t.float()) for name, t in raw.items()}
        return Prediction(outputs, valid, batch.metas, batch.items)

    def _count_bad(self, outputs: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        """Count the items with data whose outputs hold infinity or NaN, without waiting for the
        device. Items without data may give anything."""
        has_data = valid.flatten(1).any(dim=1)
        bad = torch.zeros_like(has_data)
        for x in outputs.values():
            if x.is_floating_point() and x.dtype not in _ISFINITE_DTYPES:
                x = x.float()  # torch.isfinite has no kernel for float8 and the like
            bad |= ~torch.isfinite(x).flatten(1).all(dim=1).to(bad.device)
        counts = (bad & has_data).sum(), has_data.sum()
        tally = self._tally
        self._tally = counts if tally is None else (tally[0] + counts[0], tally[1] + counts[1])

    def _warn_bad(self) -> None:
        """Warn, at the end of a run, about the items :meth:`_count_bad` found."""
        if self._tally is None:
            return
        with device_lock(self.device):
            bad, total = (int(count) for count in self._tally)
        self._tally = None
        if not bad:
            return
        dtype = str(self.dtype).removeprefix("torch.")
        advice = (
            "Check the model's outputs."
            if self.dtype == torch.float32
            else "Pass dtype='float32', which holds far larger values."
        )
        warn_caller(
            f"{bad} of {total} patches or chips with data gave infinite or NaN outputs in {dtype}, "
            f"so their pixels may be NaN or wrong. {advice}"
        )

    def _call(self, model_in: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The models on float32 (or integer) inputs, cast to the inference dtype. ``model_in``
        is emptied so the float32 copies are freed first."""
        cast = {
            n: cast_input(n, t, model_in, self.dtype, self.exact_vectors)
            for n, t in model_in.items()
        }
        model_in.clear()
        with torch.no_grad():
            return ensemble(self.models, cast[SINGLE] if SINGLE in cast else cast)

    def run(self, batches: Iterator[RawBatch], sink: Sink) -> None:
        """Predict every batch and hand the predictions to ``sink``. Errors from loading,
        the models or the sink are raised here, and background threads are stopped."""
        # a sink may move its mosaics to the host once it knows their size (mosaic_device=
        # "auto"), so a CUDA run keeps a copy stream ready and uses it while sink.to_host
        copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and (sink.to_host or getattr(sink, "auto_device", False))
            else None
        )
        pending: tuple[Prediction, Any] | None = None
        self._tally = None  # a tiled run may restart after re-planning, and a Runner is reused

        def finish_pending() -> None:
            nonlocal pending
            if pending is None:
                return
            prediction, done = pending
            pending = None
            stream = cast(torch.cuda.Stream, copy_stream)
            with torch.cuda.stream(stream):
                stream.wait_event(done)
                sink(prediction)

        try:
            with self.claim(), closing(prefetch(batches, self.device)) as prefetched:
                for batch in prefetched:
                    # Lock per batch rather than per call, so the lock isn't held while waiting on
                    # loaders, which may run MPS predictions themselves
                    devices = [
                        d for images in batch.inputs.values() for d in tensor_devices(images)
                    ]
                    with device_lock(self.device, *devices):
                        if batch.is_uint16 is None:
                            batch.dtypes = {
                                n: mixed
                                for n, i in batch.inputs.items()
                                if (mixed := chip_dtypes(i)) is not None
                            }
                            collated = {n: collate(i, self.device) for n, i in batch.inputs.items()}
                            batch.inputs = {n: c[0] for n, c in collated.items()}
                            batch.is_uint16 = {n: c[1] for n, c in collated.items()}
                        self.prepare(batch)
                        prediction = self.predict(batch)
                        if copy_stream is None or not sink.to_host:
                            # store now, so device tensors are freed before the lock is released
                            sink(prediction)
                        else:
                            done = torch.cuda.Event()
                            # on the inference device's stream, not the current device's
                            done.record(torch.cuda.current_stream(self.device))
                            # this batch is queued on the device, so finish the previous one
                            finish_pending()
                            pending = (prediction, done)
                        del batch, prediction
                    # outside the lock: a sink may wait here on work (a postprocess) that
                    # runs MPS predictions of its own
                    after_batch = getattr(sink, "after_batch", None)
                    if after_batch is not None:
                        after_batch()
                if copy_stream is not None:
                    with device_lock(self.device):
                        finish_pending()
            self._warn_bad()
        finally:
            # on errors, drop the pending batch so its device memory is freed even while the
            # caller holds the traceback
            pending = None


@dataclass
class RunSetup:
    """What every entry point works out first. ``batch_size`` and ``normalize`` are final: a
    program's batch size and the model's stored normalisation are already applied."""

    runner: Runner
    tiling: TilingSpec | None
    batch_size: int
    normalize: Any


def set_up_run(
    model: Any,
    *,
    device: str | torch.device | None,
    dtype: str | torch.dtype,
    batch_size: int,
    default_batch_size: int | None,
    spec: dict[str, Any] | None,
    normalize: Any,
    activation: Any,
    output: Any,
    nodata: float | None,
    compile: bool,
) -> RunSetup:
    """The models, device, dtype, batch size, tiling spec and normalisation of a call, and a
    :class:`Runner` for them. The runner's normalisers and primary input are set later, once the
    inputs are known. ``default_batch_size``: see :func:`program_batch_size`."""
    models = as_model_list(model)
    resolved = resolve_device(program_device(models, device))
    resolved_dtype = inference_dtype(dtype, resolved)
    batch_size = program_batch_size(models, batch_size, default=default_batch_size)
    tiling = get_tiling_spec(models, spec, normalize)
    if normalize is None and tiling:
        normalize = tiling.get("normalize")
    runner = Runner(
        models=models,
        device=resolved,
        dtype=resolved_dtype,
        normalizers={},
        nodata=nodata,
        primary=SINGLE,
        make_specs=lambda names: make_specs(
            names,
            activation,
            output,
            output_activations(tiling),
            output_names(tiling),
            regression_outputs(tiling),
        ),
        compile=compile,
        exact_vectors=vector_inputs(tiling),
    )
    return RunSetup(runner, tiling, batch_size, normalize)
