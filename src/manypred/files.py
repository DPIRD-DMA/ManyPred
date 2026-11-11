"""Files in, files out: load, predict and save many items, overlapping the three."""

from __future__ import annotations

import functools
import inspect
import math
import numbers
import os
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .batch import (
    build_normalizers,
    finish_batch,
    iter_item_batches,
    load_item,
    name_item,
    one_raster_spec,
    output_factors_of,
    primary_input,
    raster_names,
    unwrap,
)
from .blend import check_merge
from .blocks import BlockPostprocess, Scene
from .collate import SINGLE
from .inputs import describe_inputs
from .io_lock import IO_LOCK
from .lazy import is_reader, reader_profile
from .runner import (
    Prediction,
    Runner,
    program_patch_size,
    set_up_run,
    warn_caller,
    whole,
)
from .spec import (
    fit_patch,
    input_order,
    input_scales,
    named_output_factors,
    output_alias,
    output_classes,
    output_factors,
)
from .tiled import Tiler, resolve_mosaic_device

_MODES = ("tiled", "batch")
# how long a failed run waits for saves it has already queued, before raising its error
# (predict_files' docstring states it)
_SAVE_GRACE_SECONDS = 60
# default items per batch, as predict_array and predict_batch
_BATCH_SIZES = {"tiled": 1, "batch": 8}
# default bound on predictions waiting to be saved: whole scenes are large, chips small
_MAX_SAVES = {"tiled": 2, "batch": 32}


_EXTRA_KEYWORDS = {"valid", "grid", "factors"}


def _extra_keywords(save_func: Callable) -> set[str]:
    """Which of the optional ``valid``, ``grid`` and ``factors`` keywords ``save_func`` takes."""
    try:
        parameters = inspect.signature(save_func).parameters.values()
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return set(_EXTRA_KEYWORDS)
    return {p.name for p in parameters} & _EXTRA_KEYWORDS


@functools.lru_cache(maxsize=256)
def _ignores_case(directory: str) -> bool:
    """Whether the volume holding ``directory`` ignores case (macOS's default): its nearest
    existing folder found with its case swapped. True when that can't tell (no letters)."""
    path = Path(directory)
    while not path.exists() and path != path.parent:
        path = path.parent
    flipped = str(path).swapcase()
    if flipped == str(path):
        return True
    return os.path.exists(flipped) and os.path.samefile(path, flipped)


def _path_key(path: Any) -> str:
    """What two spellings of one file share. URIs (``s3://...``) are compared as written, since
    their keys aren't local paths. ``normcase`` doesn't know that macOS volumes usually ignore
    case, so the volume is asked."""
    if "://" in str(path):
        return str(path)
    resolved = Path(path).resolve()
    key = os.path.normcase(resolved)
    if sys.platform == "darwin" and _ignores_case(str(resolved.parent)):
        key = key.casefold()
    return key


class _OutputClaims:
    """Every file a run writes, refusing one that is an input or another item's output.

    Each item's output path is checked before any work (``paths``), with its ``.json`` when
    ``sidecars`` (the default writer may write either, and knows which only after the first
    batch). A dict model's head files (``<stem>_<head>.tif``) are known only once its outputs
    are: :meth:`check_heads` checks every item's file for each head the first time a writer
    is about to write one."""

    def __init__(
        self,
        items: list[Any],
        output_path: Callable[[Any, Any], Any],
        output_dir: Any,
        sidecars: bool,
    ):
        self.items = items
        self.inputs = {_path_key(item) for item in items if isinstance(item, (str, os.PathLike))}
        self.owners: dict[str, Any] = {}
        self.paths = [output_path(item, output_dir) for item in items]
        self.heads: set[str] = set()
        self.lock = threading.Lock()  # heads are checked on the save and prediction threads
        for item, path in zip(items, self.paths, strict=True):
            outputs = {_path_key(path): path}
            if sidecars:
                sidecar = Path(path).with_suffix(".json")
                outputs.setdefault(_path_key(sidecar), sidecar)
            for key, output in outputs.items():
                self._claim(key, output, item)

    def _claim(self, key: str, output: Any, item: Any) -> None:
        if key in self.owners:
            raise ValueError(
                f"{self.owners[key]} and {item} would both be written to {output}, the second "
                f"replacing the first. Pass output_path=f(item, output_dir) -> path with a "
                f"different path for each."
            )
        if key in self.inputs:
            raise ValueError(
                f"The output of {item} would be {output}, which is also an input. Pass "
                f"another output_dir, or leave earlier outputs out of items."
            )
        self.owners[key] = item

    def check_heads(self, names: Iterable[str]) -> None:
        """Claim every item's file for each head in ``names`` not seen before."""
        from .geo import head_path

        with self.lock:
            for name in names:
                if not name or name in self.heads:
                    continue
                for item, path in zip(self.items, self.paths, strict=True):
                    target = head_path(path, name)
                    self._claim(_path_key(target), target, item)
                self.heads.add(name)


def _check_output_paths(
    items: list[Any], output_path: Callable[[Any, Any], Any], output_dir: Any, sidecars: bool
) -> list[Any]:
    """Each item's output path, checked as :class:`_OutputClaims` does."""
    return _OutputClaims(items, output_path, output_dir, sidecars).paths


class _FileNodata:
    """Warns once, for ``nodata=None``, about the first item whose ``meta`` declares a nodata
    value, as those pixels are then treated as data. Only finite numbers count: NaN and infinity
    are missing anyway, and ``meta`` may hold anything for a custom ``save_func``. Call on the
    caller's thread, so the warning points at its code."""

    def __init__(self, nodata: float | None):
        self.done = nodata is not None

    def check(self, meta: Any, item: Any) -> None:
        if self.done or not isinstance(meta, Mapping):
            return
        value = meta.get("nodata")
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            return
        if not math.isfinite(value):
            return
        self.done = True
        warn_caller(
            f"{item} declares nodata={value}, but nodata=None treats those pixels as data. Pass "
            f'nodata={value} to mask them, or nodata=float("nan") to silence this.'
        )


class _LockedWriter:
    """A block writer whose writes take turns with loads and reads."""

    def __init__(self, writer: Any):
        self._writer = writer

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._writer, name)
        if name not in ("write", "close", "abort"):
            return attribute

        def locked(*args: Any, **kwargs: Any) -> Any:
            with IO_LOCK.write():
                return attribute(*args, **kwargs)

        return locked


def _writes_blocks(save_func: Any) -> bool:
    """Whether ``save_func`` opted in to writing lazy scenes in row blocks: ``writes_blocks =
    True`` and ``open(meta, item, output_dir, grid=)`` returning a writer with ``write(name,
    top, block, shape, valid, factor=)``, ``close(extras) -> result`` and optionally
    ``abort()``."""
    return getattr(save_func, "writes_blocks", False) is True


class _Saver:
    """Runs saves on one background thread, at most ``limit`` waiting; errors surface on
    the next submit or at :meth:`close`."""

    def __init__(
        self,
        save_func: Callable,
        output_dir: Any,
        limit: int,
        postprocess: Callable[[Any], Any] | None = None,
        io_lock: bool = True,
    ):
        self.io_lock = io_lock
        self.save_func = save_func
        self.postprocess = postprocess
        self.keywords = _extra_keywords(save_func)
        self.output_dir = output_dir
        self.limit = limit
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: deque[tuple[int, Future]] = deque()
        self.results: dict[int, Any] = {}

    def submit(
        self,
        index: int,
        prediction: Any,
        meta: Any,
        item: Any,
        valid: np.ndarray | None,
        grid: tuple[int, int] | None = None,
        factors: Mapping[str, int | None] | None = None,
        wait: bool = True,
    ) -> None:
        """Queue a save. ``wait=False`` skips the wait for room; the caller then calls
        :meth:`wait_for_room` itself, outside the device lock, since a ``postprocess`` that
        predicts needs that lock."""
        if wait:
            self.wait_for_room()
        extra = {"valid": valid, "grid": grid, "factors": factors}
        kwargs = {key: extra[key] for key in self.keywords}
        future = self.executor.submit(self._save, prediction, meta, item, kwargs)
        self.pending.append((index, future))

    def wait_for_room(self) -> None:
        """Block until fewer than ``limit`` saves are pending."""
        while len(self.pending) >= self.limit:
            done_index, future = self.pending.popleft()
            self.results[done_index] = future.result()

    def _save(self, prediction: Any, meta: Any, item: Any, kwargs: dict) -> Any:
        try:
            if self.postprocess is not None:
                prediction = self.postprocess(prediction)
            with IO_LOCK.write() if self.io_lock else nullcontext():
                return self.save_func(prediction, meta, item, self.output_dir, **kwargs)
        except Exception as error:
            name_item(error, item)
            raise

    def close(self, wait_for_results: bool, finish_queued: bool = False) -> None:
        """With ``wait_for_results``, wait for every save and raise the first save error.
        Otherwise an error is already propagating. With ``finish_queued``, give the saves queued
        up to ``_SAVE_GRACE_SECONDS`` to finish, as their predictions are done, and leave their
        own errors out. Then cancel what is left and return without waiting for a save that is
        running, so a hung save_func can't hide the error."""
        if not wait_for_results:
            try:
                if finish_queued:
                    futures = [future for _, future in self.pending]
                    _, running = wait(futures, timeout=_SAVE_GRACE_SECONDS)
                    if running:
                        try:
                            warn_caller(
                                f"{len(running)} saves weren't finished {_SAVE_GRACE_SECONDS} "
                                f"seconds after an error. One may still be running: let it finish "
                                f"before retrying, as it could overwrite the retry's output."
                            )
                        except Warning:
                            pass  # warnings as errors: the run's own error matters more
            finally:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.pending.clear()
            return
        self.executor.shutdown(wait=True)
        while self.pending:
            index, future = self.pending.popleft()
            self.results[index] = future.result()


def predict_files(
    model: Any,
    items: Iterable[Any],
    *,
    load_func: Callable,
    save_func: Callable | None = None,
    output_dir: str | Path | None = None,
    output_path: Callable[[Any, Any], Any] | None = None,
    overwrite: bool = True,
    mode: str = "tiled",
    max_saves_in_flight: int | None = None,
    patch_size: int = 1000,
    overlap: int = 300,
    batch_size: int | None = None,
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
    block_postprocess: BlockPostprocess | None = None,
    io_lock: bool = True,
    scales: dict[str, float] | None = None,
    spec: dict[str, Any] | None = None,
    progress: bool = True,
) -> list[Any]:
    """Load, predict and save many items. Loading the next item, predicting, and saving run on
    separate threads. Inference overlaps both, but loads and saves take turns, across the
    process, because GDAL can crash or corrupt a file otherwise (see :mod:`manypred.io_lock`).
    So ``load_func`` and ``save_func`` must not call this function.

    Args:
        model: As for :func:`~manypred.predict_array`.
        items: Items (usually file paths) passed to ``load_func`` and ``save_func``.
        load_func: ``load_func(item) -> image`` or ``(image, meta)``, called on a background
            thread (in batch mode, up to ``batch_size`` at once, so it must be thread-safe); it
            must return CPU data: arrays, tensors, or lazy readers (see :mod:`manypred.lazy`;
            batch mode reads a reader whole). ``meta`` is passed to ``save_func`` (a rasterio
            profile for the default GeoTIFF writer); without one, a reader's ``profile`` is used.
        save_func: ``save_func(prediction, meta, item, output_dir) -> path``, called on a background
            thread with the item's array (or dict of arrays). If it accepts a ``valid`` keyword it
            also gets the (H, W) bool valid-pixel mask (None without ``mask_nodata``), with ``grid``
            the (H, W) the model predicted on, and with ``factors`` ``{output: f}`` for every
            output: how many times coarser than ``grid`` each dense output is (1 at full
            resolution), None for image-level outputs; a bare tensor's output is ``""`` (all three
            for ``**kwargs``; the same in both modes). Default: :class:`~manypred.geo.GeoTIFFWriter`
            (``[geo]`` extra), which writes ``output_path(item, output_dir)`` with the mask. A
            save_func with ``writes_blocks = True`` is also a block writer: lazy scenes call its
            ``open(meta, item, output_dir, grid=)`` instead, and write rows to the returned object's
            ``write(name, top, block, shape, valid, factor=)``, then call its ``close(extras)``
            (whose result is the item's) or, on an error, its ``abort()`` if it has one.
        output_dir: Passed to ``save_func`` and ``output_path``; default None, next to each
            item.
        output_path: ``output_path(item, output_dir) -> path`` of an item's output, used to
            skip items with ``overwrite=False``, and by the default writer. Default:
            :func:`~manypred.geo.default_output_path` (``<stem>_pred.tif``). It must give the same
            path every time for an item. Raises ValueError before any work if two items share a
            path or a path is also an input.
        overwrite: If False, items whose ``output_path`` exists are skipped without loading.
        mode: ``"tiled"`` (each item is a scene, predicted in patches and blended) or
            ``"batch"`` (each item is a chip, predicted whole, ``batch_size`` chips at a time).
        max_saves_in_flight: Most predictions waiting to be saved, so a slow disk can't make
            them pile up in memory. Default 2 scenes in tiled mode, 32 chips in batch mode.
        patch_size, overlap: Tiled mode only; as for :func:`~manypred.predict_array`.
        mosaic_device, mosaic_dtype: Tiled mode only; as for :func:`~manypred.predict_array`.
        shift_patches, adapt_patch_size, merge: Tiled mode only; as for
            :func:`~manypred.predict_array`.
        io_lock: Whether ``load_func`` and ``save_func`` take turns (see above). False stops a slow
            one that doesn't use GDAL holding up the other, which can leave the GPU idle. If either
            calls GDAL directly, such as through rasterio, keep True: the crash this prevents is
            back otherwise. ManyPred's own helpers (:func:`~manypred.geo.load_raster`,
            :func:`~manypred.geo.stack_rasters`, :class:`~manypred.geo.RasterioReader`,
            :func:`~manypred.geo.write_geotiff`) and the default writer take turns either way.
        scales, spec: As for :func:`~manypred.predict_array`.
        postprocess: ``f(prediction) -> anything``, run on the save thread before
            ``save_func``, which then receives its result.
        block_postprocess: Tiled mode only. A :class:`~manypred.blocks.BlockPostprocess`
            run on each scene's stitched outputs block by block (core plus halo), for
            post-processing too large for a whole scene, such as decoding vectors. Its
            results go to its ``save`` or its ``results`` attribute; this function returns
            what it would without it, and the rasters are saved as usual. It sees the
            outputs before ``postprocess`` and before output conversion (argmax). Lazy
            scenes written in row blocks are post-processed while they are predicted,
            holding a strip of rows; other scenes once stitched.
        batch_size: Patches or chips per batch. Default 1 in tiled mode, as for
            :func:`~manypred.predict_array`, and 8 in batch mode, as for
            :func:`~manypred.predict_batch`.
        device, dtype, compile: As for :func:`~manypred.predict_array` and
            :func:`~manypred.predict_batch`.
        normalize, activation, output: As for :func:`~manypred.predict_array`.
        nodata, mask_nodata, progress: As for :func:`~manypred.predict_array`. With
            ``nodata=None``, a nodata value declared in an item's ``meta`` isn't used, and gives
            one warning per call; ``nodata=float("nan")`` silences it.

    Returns:
        One entry per item, in order: what ``save_func`` returned, or ``output_path`` for
        skipped items.

    An error stops the run. Its traceback names the item if it came from loading, saving or a
    scene in tiled mode, but not from a batch of chips in the model. Saves already queued get up
    to 60 seconds to finish before the error is raised (with a warning if they don't), and an
    interrupt doesn't wait, so a slow save may still be running when the error arrives. After
    fixing or removing that item, ``overwrite=False`` resumes, predicting whatever has no output
    yet.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    if IO_LOCK.held():
        # its saves would wait for the load or save this call is running inside
        raise RuntimeError("predict_files can't run inside a load_func or save_func.")
    asked_batch_size = batch_size
    if batch_size is None:
        batch_size = _BATCH_SIZES[mode]
    batch_size = whole("batch_size", batch_size, minimum=1)
    if max_saves_in_flight is not None:
        max_saves_in_flight = whole("max_saves_in_flight", max_saves_in_flight, minimum=1)
    patch_size, overlap = whole("patch_size", patch_size), whole("overlap", overlap)
    check_merge(merge)
    if block_postprocess is not None and mode != "tiled":
        raise ValueError("block_postprocess needs mode='tiled'.")
    if block_postprocess is not None:
        block_postprocess.results = {}  # this call's results only
    items = list(items)
    existing = None
    writer = None
    if save_func is None:
        from .geo import GeoTIFFWriter, _rasterio, default_output_path

        _rasterio()  # without it, fail now rather than at the first save
        output_path = output_path or default_output_path
        save_func = writer = GeoTIFFWriter(output_path)
        existing = writer.existing
    elif not overwrite and output_path is None:
        raise ValueError(
            "overwrite=False with a custom save_func needs output_path=f(item, output_dir) "
            "-> path, to tell which items are already done."
        )
    paths = None
    if output_path is not None:
        # name every output now, so items that can't be named fail before any work
        claims = _OutputClaims(items, output_path, output_dir, sidecars=existing is not None)
        paths = claims.paths
        if writer is not None:
            writer.check_heads = claims.check_heads

    results: list[Any] = [None] * len(items)
    todo = []
    for index, item in enumerate(items):
        if not overwrite:
            if existing is not None:
                path = existing(item, output_dir)
            else:
                assert paths is not None
                path = paths[index] if Path(paths[index]).exists() else None
            if path is not None:
                results[index] = path
                continue
        todo.append(index)

    setup = set_up_run(
        model,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        default_batch_size=batch_size if asked_batch_size is None else None,
        spec=spec,
        normalize=normalize,
        activation=activation,
        output=output,
        nodata=nodata,
        compile=compile,
    )
    runner, tiling, device = setup.runner, setup.tiling, setup.runner.device
    batch_size, normalize = setup.batch_size, setup.normalize
    if writer is not None and postprocess is None:  # what postprocess returns isn't these outputs
        writer.classes = {
            name: names
            for name, names in output_classes(tiling).items()
            if not callable(
                output.get(name, output.get("")) if isinstance(output, Mapping) else output
            )
        }
        writer.input_bands = (tiling or {}).get("bands")
    if mode == "tiled":
        patch_size, overlap = fit_patch(tiling, patch_size, overlap)
        patch_size, overlap = program_patch_size(
            runner.models, patch_size, overlap, adapt_patch_size
        )
    saver = _Saver(
        save_func, output_dir, max_saves_in_flight or _MAX_SAVES[mode], postprocess, io_lock
    )
    pbar = tqdm(
        total=len(items),
        initial=len(items) - len(todo),
        disable=not progress,
        desc=f"Predicting on {device.type} {str(runner.dtype).split('.')[-1]}",
    )
    ok = finish_queued = False
    try:
        if todo:
            if mode == "tiled":
                tiler = Tiler(
                    runner,
                    patch_size,
                    overlap,
                    batch_size,
                    resolve_mosaic_device(mosaic_device, device),
                    mask_nodata,
                    progress=False,
                    normalize=normalize,
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
                _run_tiled(
                    tiler,
                    items,
                    todo,
                    load_func,
                    saver,
                    pbar,
                    block_postprocess,
                    _FileNodata(nodata),
                    io_lock,
                )
            else:
                _run_batch(
                    runner,
                    items,
                    todo,
                    load_func,
                    saver,
                    pbar,
                    batch_size,
                    normalize,
                    mask_nodata,
                    _FileNodata(nodata),
                    io_lock,
                    input_order(tiling),
                    named_output_factors(tiling),
                )
        ok = True
    except Exception:
        finish_queued = True  # not on an interrupt: then the caller wants to stop now
        raise
    finally:
        try:
            saver.close(wait_for_results=ok, finish_queued=finish_queued)
        finally:
            # even when a save failed: the block workers are processes
            if block_postprocess is not None:
                block_postprocess.close(cancel=not ok)
            pbar.close()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for index, value in saver.results.items():
        results[index] = value
    return results


def _scene(meta: Any, grid: tuple[int, int]) -> Scene:
    """The :class:`~manypred.blocks.Scene` of the grid predicted on: with the transform and
    CRS of ``meta`` when it is a rasterio profile of that grid."""
    if not isinstance(meta, Mapping):
        return Scene(*grid)
    size = (meta.get("height"), meta.get("width"))
    if size != (None, None) and size != tuple(grid):  # the profile of another grid
        return Scene(*grid)
    return Scene(grid[0], grid[1], meta.get("transform"), meta.get("crs"))


def _run_tiled(
    tiler: Tiler,
    items: list,
    todo: list[int],
    load_func: Callable,
    saver: _Saver,
    pbar: tqdm,
    post: BlockPostprocess | None = None,
    file_nodata: _FileNodata | None = None,
    io_lock: bool = True,
) -> None:
    """Scenes one at a time; the next scene loads while the current one is predicted."""
    loader = ThreadPoolExecutor(max_workers=1)
    next_load = loader.submit(load_item, items[todo[0]], load_func, io_lock)
    warned_whole = False
    ok = False
    try:
        for position, index in enumerate(todo):
            try:
                inputs, meta = next_load.result()
                if position + 1 < len(todo):
                    next_load = loader.submit(
                        load_item, items[todo[position + 1]], load_func, io_lock
                    )
                scene = inputs[SINGLE] if set(inputs) == {SINGLE} else inputs
                lazy = any(is_reader(x) for x in inputs.values())
                primary, _ = describe_inputs(inputs, tiler.scales, tiler.order)
                if meta is None:
                    meta = reader_profile(inputs[primary])
                if file_nodata is not None:
                    file_nodata.check(meta, items[index])
                grid = tuple(inputs[primary].shape[-2:])
                where = _scene(meta, grid) if post is not None else None
                if lazy and _writes_blocks(saver.save_func) and saver.postprocess is None:
                    # a lazy scene: write finished rows as they come, never the whole output
                    locked = saver.io_lock or getattr(saver.save_func, "uses_gdal", False)
                    with IO_LOCK.write() if locked else nullcontext():
                        opened = saver.save_func.open(
                            meta, items[index], saver.output_dir, grid=grid
                        )
                    writer = _LockedWriter(opened) if locked else opened
                    stream = None
                    try:
                        stream = post.stream(where) if post is not None else None
                        extras = tiler.predict_blocks(scene, writer, hook=stream)
                        if stream is not None:
                            # before the writer closes, which marks the item done
                            post.finish(index, items[index], stream.close(), stream.scene)
                        saver.results[index] = writer.close(extras)
                    except BaseException:
                        if hasattr(writer, "abort"):
                            writer.abort()
                        if stream is not None:
                            stream.abort()
                        raise
                else:
                    if lazy and not warned_whole:
                        warned_whole = True
                        cause = (
                            "postprocess" if saver.postprocess is not None else "a custom save_func"
                        )
                        warn_caller(
                            f"{items[index]} is read lazily, but its output is held in memory "
                            f"whole because of {cause}. To write it in row blocks, use a "
                            f"save_func with writes_blocks=True (the default one) and no "
                            f"postprocess; block_postprocess can take postprocess's place "
                            f"without stopping that."
                        )
                    select = post.select if post is not None else None
                    result, valid, fields = tiler.predict_fields(scene, select)
                    if post is not None:
                        # before saving, which marks the item done
                        post.finish(index, items[index], post.run(fields, where), where)
                    saver.submit(
                        index,
                        unwrap(result),
                        meta,
                        items[index],
                        valid,
                        grid,
                        tiler.last_factors,
                    )
                pbar.update(1)
            except Exception as error:
                name_item(error, items[index])
                raise
        ok = True
    finally:
        # on errors, don't wait for a load that is running: it may be blocked on I/O
        loader.shutdown(wait=ok, cancel_futures=True)


class _SaveSink:
    """Batch-mode sink: finishes each chip and hands it to the saver."""

    to_host = True

    def __init__(
        self,
        runner: Runner,
        saver: _Saver,
        indexes: Iterator[int],
        items: list,
        mask: bool,
        pbar: tqdm,
        file_nodata: _FileNodata | None = None,
        factors: Mapping[str, int] | None = None,
    ):
        self.runner, self.saver, self.indexes = runner, saver, indexes
        self.items, self.mask, self.pbar = items, mask, pbar
        self.file_nodata = file_nodata
        self.factors = factors or None

    def __call__(self, prediction: Prediction) -> None:
        assert self.runner.specs is not None
        outputs = {
            name: x.numpy(force=True)
            for name, x in finish_batch(
                prediction, self.runner.specs, self.mask, self.factors
            ).items()
        }
        valid = prediction.valid[:, 0].numpy(force=True) if self.mask else None
        grid = tuple(prediction.valid.shape[-2:])
        # as tiled mode gives them: every output, None for image-level ones
        factors = output_factors_of(prediction.outputs, grid, self.factors)
        for i, meta in enumerate(prediction.metas):
            index = next(self.indexes)
            if self.file_nodata is not None:
                self.file_nodata.check(meta, self.items[index])
            chip = unwrap({name: x[i] for name, x in outputs.items()})
            self.saver.submit(
                index,
                chip,
                meta,
                self.items[index],
                valid[i] if valid is not None else None,
                grid,
                factors,
                wait=False,
            )
            self.pbar.update(1)

    def after_batch(self) -> None:
        """Called by the runner after each batch, with the device lock released."""
        self.saver.wait_for_room()


def _run_batch(
    runner: Runner,
    items: list,
    todo: list[int],
    load_func: Callable,
    saver: _Saver,
    pbar: tqdm,
    batch_size: int,
    normalize: Any,
    mask_nodata: bool,
    file_nodata: _FileNodata | None = None,
    io_lock: bool = True,
    order: Sequence[str] = (),
    factors: Mapping[str, int] | None = None,
) -> None:
    """Chips in batches; ``factors``: the dense outputs' reductions the spec declares (the
    others are inferred from the sizes)."""
    batches = iter_item_batches(
        (items[i] for i in todo),
        load_func,
        batch_size,
        shape_hint="Resize them in load_func, or use mode='tiled'.",
        io_lock=io_lock,
        order=order,
    )
    first = next(batches)
    runner.normalizers = build_normalizers(
        one_raster_spec(normalize, raster_names(first)), list(first.inputs), batch_mode=True
    )
    runner.primary = primary_input(first, order)

    def all_batches():
        yield first
        yield from batches

    runner.run(
        all_batches(),
        _SaveSink(runner, saver, iter(todo), items, mask_nodata, pbar, file_nodata, factors),
    )
