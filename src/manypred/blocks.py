"""Block post-processing over stitched rasters: a fixed grid of blocks with a halo.

A scene's stitched outputs (for example GeoTIFFs written by :func:`manypred.predict_files`
with ``merge="cut"``) are too large to post-process whole, say to decode vectors from them.
:func:`postprocess_blocks` calls a function on one block at a time: its core, a tile of the
grid, grown by ``halo`` pixels on every side (clipped to the scene), so the function sees the
context around the core. The cores partition the scene, so a function that keeps only what it
finds in the core (and drops what touches the window's cut edges) can give the same result as
one run over the whole scene. The grid depends only on the scene size, not on how the scene was
tiled for prediction.

Fields are arrays (or anything sliced like one: memmaps, zarr) or paths to raster files read
lazily, window by window, with rasterio (the ``[geo]`` extra), so only one window per field is
in memory per block (per worker).
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import os
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .io_lock import IO_LOCK
from .windows import Window


@dataclass(frozen=True)
class Block:
    """One block of a :func:`block_grid`.

    Attributes:
        index: Row-major position in the grid.
        core: ``(top, bottom, left, right)``, half-open, in scene pixels; the cores of a grid
            partition the scene.
        window: The core grown by the halo on every side, clipped to the scene.
        scene_edges: Whether the window's top, bottom, left and right sides are the scene's
            edge (rather than a cut through the scene).
    """

    index: int
    core: Window
    window: Window
    scene_edges: tuple[bool, bool, bool, bool]

    @property
    def inner(self) -> Window:
        """The core in window coordinates, so ``fields[:, top:bottom, left:right]`` of a window
        is the core."""
        top, bottom, left, right = self.core
        return (
            top - self.window[0],
            bottom - self.window[0],
            left - self.window[2],
            right - self.window[2],
        )


@dataclass(frozen=True)
class Scene:
    """The scene the blocks tile: its size in pixels and, if known, the affine transform of
    its pixel grid (a rasterio ``Affine``) and its CRS."""

    height: int
    width: int
    transform: Any = None
    crs: Any = None


def block_grid(height: int, width: int, block: int = 4096, halo: int = 256) -> list[Block]:
    """Row-major grid of ``block`` x ``block`` cores over a ``height`` x ``width`` scene; the
    last row and column of cores are cut short at the scene edge (a block larger than the scene
    gives one block). Each window is its core grown by ``halo`` on every side, clipped to the
    scene."""
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}.")
    if halo < 0:
        raise ValueError(f"halo must be at least 0, got {halo}.")
    if height < 0 or width < 0:
        raise ValueError(f"The scene size must not be negative, got {(height, width)}.")
    blocks = []
    for top in range(0, height, block):
        bottom = min(top + block, height)
        for left in range(0, width, block):
            right = min(left + block, width)
            window = (
                max(top - halo, 0),
                min(bottom + halo, height),
                max(left - halo, 0),
                min(right + halo, width),
            )
            edges = (window[0] == 0, window[1] == height, window[2] == 0, window[3] == width)
            blocks.append(Block(len(blocks), (top, bottom, left, right), window, edges))
    return blocks


@dataclass(frozen=True)
class _Raster:
    """A raster file, read one window at a time (bands as channels)."""

    path: str

    def read(self, window: Window) -> np.ndarray:
        from rasterio.windows import Window as RioWindow

        from .geo import _rasterio

        top, bottom, left, right = window
        # in a worker process this lock is that process's own, as is its GDAL block cache
        with IO_LOCK.read(), _rasterio().open(self.path) as src:
            return src.read(window=RioWindow(left, top, right - left, bottom - top))


@dataclass(frozen=True)
class _Memmap:
    """A numpy memmap reopened by file name, so worker processes read their own windows
    instead of receiving them."""

    filename: str
    dtype: Any
    offset: int
    shape: tuple[int, ...]
    order: str

    @classmethod
    def of(cls, x: Any) -> _Memmap | None:
        """For a memmap over a whole mapping (not a view into one, whose offset and shape
        would not say where its data is), else None."""
        if not isinstance(x, np.memmap) or x.filename is None or isinstance(x.base, np.ndarray):
            return None
        order = "F" if x.flags.f_contiguous and not x.flags.c_contiguous else "C"
        return cls(x.filename, x.dtype, x.offset, x.shape, order)

    def read(self, window: Window) -> np.ndarray:
        x = np.memmap(self.filename, self.dtype, "r", self.offset, self.shape, self.order)
        return _cut(x, window)


def _is_path(x: Any) -> bool:
    return isinstance(x, (str, os.PathLike))


def _cut(source: Any, window: Window) -> np.ndarray:
    """An array-like under ``window`` as a (C, h, w) array of its own (not a view)."""
    top, bottom, left, right = window
    if len(source.shape) == 2:
        return np.array(source[top:bottom, left:right])[None]
    return np.array(source[:, top:bottom, left:right])


def _inputs(sources: Mapping[str, Any], block: Block) -> dict[str, Any]:
    """A block's inputs: array windows cut, files (rasters, memmaps) left to
    :func:`_run_block`, which may run in a worker process."""
    return {
        name: source if isinstance(source, (_Raster, _Memmap)) else _cut(source, block.window)
        for name, source in sources.items()
    }


def _run_block(fn: Callable, inputs: Mapping[str, Any], block: Block, scene: Scene) -> Any:
    fields = {
        name: x.read(block.window) if isinstance(x, (_Raster, _Memmap)) else x
        for name, x in inputs.items()
    }
    return fn(fields, block, scene)


def _shape(name: str, source: Any) -> tuple[int, int]:
    shape = tuple(source.shape)
    if len(shape) not in (2, 3):
        raise ValueError(f"Field {name!r} must be (C, H, W) or (H, W), found shape {shape}.")
    return shape[-2], shape[-1]


def _sources(fields: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, int], Scene | None]:
    """Fields as array-likes or :class:`_Raster`, their common (H, W), and the scene of the
    first raster file (None without one)."""
    if not fields:
        raise ValueError("postprocess_blocks needs at least one field.")
    sources: dict[str, Any] = {}
    shapes: dict[str, tuple[int, int]] = {}
    first: Scene | None = None
    first_name = ""
    for name, field in fields.items():
        if _is_path(field):
            from .geo import _rasterio

            path = os.fspath(field)
            with IO_LOCK.read(), _rasterio().open(path) as src:
                shapes[name] = (src.height, src.width)
                if first is None:
                    first, first_name = Scene(src.height, src.width, src.transform, src.crs), name
                elif src.crs != first.crs or not src.transform.almost_equals(first.transform):
                    raise ValueError(
                        f"Raster fields must share a grid: {name!r} has transform "
                        f"{tuple(src.transform)[:6]}, CRS {src.crs}; {first_name!r} has "
                        f"{tuple(first.transform)[:6]}, CRS {first.crs}."
                    )
            sources[name] = _Raster(path)
        else:
            if not hasattr(field, "shape"):
                raise TypeError(
                    f"Field {name!r} must be an array-like with .shape (sliced like numpy) or a "
                    f"path to a raster file, got {type(field).__name__}."
                )
            shapes[name] = _shape(name, field)
            sources[name] = _Memmap.of(field) or field
    if len(set(shapes.values())) > 1:
        raise ValueError(f"All fields must have the same height and width, got {shapes}.")
    return sources, next(iter(shapes.values())), first


def postprocess_blocks(
    fields: Mapping[str, Any],
    fn: Callable[[dict[str, np.ndarray], Block, Scene], Any],
    *,
    block: int = 4096,
    halo: int = 256,
    combine: Callable[[list[Any], Scene], Any] | None = None,
    workers: int = 0,
    scene: Scene | None = None,
) -> Any:
    """Call ``fn`` on every block of a stitched scene, in grid order.

    Args:
        fields: ``{name: field}``, each a ``(C, H, W)`` or ``(H, W)`` array-like that slices
            like numpy (ndarray, memmap, zarr array), or a path to a raster file (bands as
            channels; e.g. a GeoTIFF written by :func:`manypred.predict_files`), read lazily,
            one window per block. All fields must share ``(H, W)``, and raster files their transform
            and CRS.
        fn: ``fn(window_fields, block, scene) -> result``, where ``window_fields`` is ``{name:
            (C, h, w) numpy array}`` (always 3-D, in the field's dtype, a copy) cropped to
            ``block.window``, ``block`` a :class:`Block` and ``scene`` a :class:`Scene`.
        block: Core size in pixels; see :func:`block_grid`.
        halo: Pixels of context added around each core (clipped at the scene edge).
        combine: ``combine(results, scene) -> anything``, called once with the list of
            results in grid order; its value is returned. Default: return the list.
        workers: 0 (the default) runs the blocks here, one at a time. More runs them in a pool
            of that many processes: raster files and memmaps are read in the workers; other
            array windows are cut here and sent, one block ahead of the workers, so about two
            windows per worker are in memory. ``fn``, its results and the fields' windows
            must then be picklable, and ``fn`` importable in a fresh process (a module-level
            function or a picklable object; workers start with "spawn", so a script that
            calls this needs an ``if __name__ == "__main__":`` guard). Each worker uses one
            compute thread (torch, OpenMP, BLAS). Results keep grid order.
        scene: The scene passed to ``fn`` and ``combine``. Default: the size, transform and
            CRS of the first field that is a raster file, else ``Scene(H, W)``. A given scene
            must match the fields' size.

    Returns:
        ``combine(results, scene)``, or the list of results in grid order.
    """
    if workers < 0:
        raise ValueError(f"workers must be at least 0, got {workers}.")
    sources, (height, width), first = _sources(fields)
    if scene is None:
        scene = first if first is not None else Scene(height, width)
    elif (scene.height, scene.width) != (height, width):
        raise ValueError(
            f"scene is {scene.height} x {scene.width}, but the fields are {height} x {width}."
        )
    return _run_grid(fn, sources, block, halo, combine, scene, workers)


def _run_grid(
    fn: Callable,
    sources: Mapping[str, Any],
    block: int,
    halo: int,
    combine: Callable | None,
    scene: Scene,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> Any:
    """Every block of ``scene`` through ``fn``, then ``combine``; in a pool of ``workers``
    processes (``executor``, or one made for this call) when ``workers > 0``."""
    depth = None if _all_files(sources) else workers + 1
    runner = _Runner(fn, scene, workers, depth, executor)
    try:
        for b in block_grid(scene.height, scene.width, block, halo):
            runner.submit(_inputs(sources, b), b)
    except BaseException:
        runner.abort()
        raise
    results = runner.finish()
    return combine(results, scene) if combine is not None else results


def _all_files(sources: Mapping[str, Any]) -> bool:
    return all(isinstance(s, (_Raster, _Memmap)) for s in sources.values())


# environment variables that size the thread pools of OpenMP, BLAS libraries and torch
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


_ENVIRON_LOCK = threading.Lock()


class _OneThreadProcess(multiprocessing.context.SpawnProcess):
    """A spawned process that starts with the thread-count variables at 1: they are set in
    this process's environment only while the child is launched, so the child has them before
    it imports anything (numpy, BLAS and torch size their thread pools when imported, which
    under spawn happens before a pool initializer runs)."""

    def start(self) -> None:
        with _ENVIRON_LOCK:
            saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
            os.environ.update(dict.fromkeys(_THREAD_VARIABLES, "1"))
            try:
                super().start()
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


class _SpawnContext(multiprocessing.context.SpawnContext):
    Process = _OneThreadProcess


def _init_worker() -> None:
    """``torch.set_num_threads(1)`` if torch is already imported in the worker (by the
    parent's main module, which spawn re-imports), in case it was sized before the variables
    applied."""
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def make_pool(workers: int) -> ProcessPoolExecutor:
    """A pool of ``workers`` processes for blocks: started with "spawn", so a parent that has
    started threads (torch, OpenMP, BLAS, CUDA) doesn't hang forked workers (the default on
    Linux before Python 3.14), with one compute thread each (:class:`_OneThreadProcess`,
    :func:`_init_worker`)."""
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=_SpawnContext(), initializer=_init_worker
    )


class _Runner:
    """Runs blocks here (``workers=0``) or in a process pool (``executor``, or one of its own
    from :func:`make_pool`), results in submission (grid) order. At most ``depth`` blocks are
    pending in the pool (default two per worker); array windows are held until their block is
    done, so with array inputs one block waits beyond those running (``depth=workers + 1``)."""

    def __init__(
        self,
        fn: Callable,
        scene: Scene,
        workers: int,
        depth: int | None = None,
        executor: ProcessPoolExecutor | None = None,
    ):
        self.fn, self.scene = fn, scene
        self.results: list[Any] = []
        self.pending: deque[Future] = deque()
        self.own = executor is None and workers > 0  # shut down when done
        self.executor = make_pool(workers) if self.own else executor if workers else None
        self.depth = depth or 2 * workers

    def submit(self, inputs: Mapping[str, Any], block: Block) -> None:
        if self.executor is None:
            self.results.append(_run_block(self.fn, inputs, block, self.scene))
            return
        if len(self.pending) >= self.depth:
            self.results.append(self.pending.popleft().result())
        self.pending.append(self.executor.submit(_run_block, self.fn, inputs, block, self.scene))

    def finish(self) -> list[Any]:
        """Wait for every block; the results in grid order. Errors from ``fn`` are raised
        here (or by a later :meth:`submit`)."""
        try:
            while self.pending:
                self.results.append(self.pending.popleft().result())
        except BaseException:
            self.abort()
            raise
        if self.own:
            self.executor.shutdown(wait=True)
        return self.results

    def abort(self) -> None:
        """After an error: cancel queued blocks without waiting for running ones."""
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        if self.own:
            self.executor.shutdown(wait=False, cancel_futures=True)


class BlockPostprocess:
    """Block post-processing while :func:`manypred.predict_files` predicts, so one call
    writes, say, vectors: :func:`postprocess_blocks`'s loop run on each scene's stitched
    outputs, without saving and re-reading them.

    Pass it as ``predict_files(..., block_postprocess=BlockPostprocess(fn, ...))``. For each
    scene predicted (not for items skipped with ``overwrite=False``), ``fn(window_fields,
    block, scene)`` is called on every block of :func:`block_grid` in grid order, then
    ``combine(results, scene)`` once (default: the list of results), then ``save(result, item,
    scene)`` with that. Without ``save`` the result is kept in :attr:`results`. Rasters are
    saved as ``save_func`` says, as without the hook. ``scene`` has the predicted grid's size
    and, when ``load_func``'s ``meta`` is a rasterio profile of that grid, its transform and
    CRS.

    ``window_fields`` holds the stitched outputs after merging (``merge``), activation and
    nodata masking, before any output conversion (``"argmax"`` or a callable ``output`` give
    the hook the probabilities or values they were computed from), as float32 ``(C, h, w)``
    arrays, NaN at nodata pixels with ``mask_nodata=True`` (the default; without it, nodata
    pixels a patch covered keep the model's values). ``postprocess`` does not change them; it
    only changes what is saved. For a scene read lazily and written in row blocks, the outputs
    are never held whole: a strip of block rows plus ``2 * halo`` rows is kept per field
    (``(block + 2 * halo) * W * C * 4`` bytes), and each strip's blocks run as soon as its rows
    are final. Other scenes are stitched in memory, then run through
    :func:`postprocess_blocks`. Both give the same results as :func:`postprocess_blocks` over
    the saved rasters (for outputs saved as float32 values).

    With ``workers=0`` the blocks run on the prediction thread, between batches; with
    ``workers > 0`` they run in that many processes while prediction continues (``fn`` must
    then be importable and picklable, as for :func:`postprocess_blocks`). The pool is started
    on first use and kept for the rest of the :func:`~manypred.predict_files` call, which
    closes it (:meth:`close`).

    Args:
        fn: ``fn(window_fields, block, scene) -> result``, as for :func:`postprocess_blocks`.
        block, halo, combine, workers: As for :func:`postprocess_blocks`.
        save: ``save(result, item, scene) -> anything``, called on the prediction thread once
            per scene with ``combine``'s result (e.g. to write a GeoPackage), before the
            scene's rasters are marked done: if ``fn``, ``combine`` or ``save`` fails, the
            item is not skipped by a rerun with ``overwrite=False``.
        fields: The outputs ``fn`` gets, by name; ``fn`` gets them under these names. A
            model returning a bare tensor has one output, ``""``, which may also be named as
            its tiling spec names its only output (``fn`` then gets it under that name).
            Default: every dense output, under its own name (``""`` for a bare tensor). They
            must be at full resolution; a reduced-resolution head can still be saved, but not
            passed to the hook.

    Attributes:
        results: ``{index of the item in items: combine's result, or what save returned}``,
            for the scenes post-processed in the last :func:`~manypred.predict_files` call
            (cleared when a call starts).
    """

    def __init__(
        self,
        fn: Callable[[dict[str, np.ndarray], Block, Scene], Any],
        *,
        block: int = 4096,
        halo: int = 256,
        combine: Callable[[list[Any], Scene], Any] | None = None,
        workers: int = 0,
        save: Callable[[Any, Any, Scene], Any] | None = None,
        fields: list[str] | None = None,
    ):
        block_grid(1, 1, block, halo)  # checks block and halo
        if workers < 0:
            raise ValueError(f"workers must be at least 0, got {workers}.")
        self.fn, self.block, self.halo = fn, block, halo
        self.combine, self.workers, self.save = combine, workers, save
        self.fields = None if fields is None else list(fields)
        self.results: dict[int, Any] = {}
        self._given: dict[str, str] = {}  # {output: the name fn gets it under}, see select
        self._executor: ProcessPoolExecutor | None = None

    def pool(self) -> ProcessPoolExecutor | None:
        """The worker pool, started on first use (None with ``workers=0``)."""
        if self.workers and self._executor is None:
            self._executor = make_pool(self.workers)
        return self._executor

    def close(self, cancel: bool = False) -> None:
        """Shut the worker pool down (``cancel``: without waiting, after an error). A later
        scene starts a new one."""
        if self._executor is not None:
            executor, self._executor = self._executor, None
            executor.shutdown(wait=not cancel, cancel_futures=cancel)

    def select(self, factors: Mapping[str, int | None], alias: str | None = None) -> list[str]:
        """The outputs the hook gets, from ``{output: factor, or None for an image-level
        one}``. ``alias``: the tiling spec's name for a model's only output when it returns a
        bare tensor (``""``), which ``fields`` may use, as ``merge`` and ``output`` may."""
        dense = [name for name, f in factors.items() if f is not None]
        names = dense if self.fields is None else self.fields
        # fn gets each field under the name fields= gives it (a bare tensor's output may be
        # named as the spec names it); internally it is the output's own name
        self._given = {"": alias} if alias and list(factors) == [""] and alias in names else {}
        names = ["" if self._given and name == alias else name for name in names]
        unknown = [n for n in names if n not in factors]
        if unknown:
            raise ValueError(
                f"BlockPostprocess fields {unknown} are not outputs of the model; outputs: "
                f"{list(factors)}."
            )
        image = [n for n in names if factors[n] is None]
        if image:
            raise ValueError(
                f"BlockPostprocess fields {image} are image-level outputs (one value per "
                f"patch); fields must be dense."
            )
        reduced = [n for n in names if factors[n] != 1]
        if reduced:
            full = [n for n in dense if factors[n] == 1]
            raise ValueError(
                f"BlockPostprocess fields must be at full resolution, but {reduced} are "
                f"reduced-resolution heads. Pass fields= with full-resolution outputs (here "
                f"{full}); the others are still saved."
            )
        if not names:
            raise ValueError("BlockPostprocess has no fields: the model has no dense output.")
        return names

    def run(self, fields: Mapping[str, Any], scene: Scene) -> Any:
        """The hook on a scene stitched in memory."""
        fields = self.given(fields)
        sources, size, _ = _sources(fields)
        if size != (scene.height, scene.width):
            raise ValueError(f"scene is {scene.height} x {scene.width}, the fields are {size}.")
        return _run_grid(
            self.fn, sources, self.block, self.halo, self.combine, scene, self.workers, self.pool()
        )

    def given(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """``fields`` under the names ``fn`` gets them by (see :meth:`select`)."""
        return {self._given.get(name, name): value for name, value in fields.items()}

    def stream(self, scene: Scene) -> BlockStream:
        """The hook on a scene whose outputs arrive in row blocks."""
        return BlockStream(self, scene)

    def finish(self, index: int, item: Any, result: Any, scene: Scene) -> None:
        """Hand one scene's result to ``save``, or keep it in :attr:`results`."""
        self.results[index] = self.save(result, item, scene) if self.save is not None else result


class BlockStream:
    """One scene's :class:`BlockPostprocess`, fed rows as they are finished. The rows of a
    strip (one block row of the grid, its windows' rows included) are kept in one buffer per
    field; when the strip's last window row arrives its blocks run, and the rows the next
    strip's windows share (``2 * halo``) move to the front of the buffer."""

    def __init__(self, post: BlockPostprocess, scene: Scene):
        self.post, self.scene = post, scene
        self.names: list[str] | None = None  # chosen by select() once the outputs are known
        strips: dict[int, list[Block]] = {}
        for b in block_grid(scene.height, scene.width, post.block, post.halo):
            strips.setdefault(b.core[0], []).append(b)
        # (window top, window bottom, blocks) per block row
        self.strips = [(row[0].window[0], row[0].window[1], row) for row in strips.values()]
        self.capacity = max((bottom - top for top, bottom, _ in self.strips), default=0)
        self.next = 0  # the next strip to run
        self.start = self.end = 0  # scene rows [start, end) are in the buffers
        self.buffers: dict[str, np.ndarray] = {}
        self.runner = _Runner(post.fn, scene, post.workers, post.workers + 1, post.pool())

    def select(self, factors: Mapping[str, int | None], alias: str | None = None) -> list[str]:
        """Choose the fields from the outputs' factors (see :meth:`BlockPostprocess.select`)."""
        self.names = self.post.select(factors, alias)
        return self.names

    def rows(self, top: int, fields: Mapping[str, np.ndarray]) -> None:
        """Scene rows [top, top + h) of every field, ``(C, h, W)`` each, rows in order."""
        if top != self.end:
            raise AssertionError(f"rows from {top} after rows up to {self.end}")
        assert self.names is not None and set(fields) == set(self.names)
        height = next(iter(fields.values())).shape[1]
        done = 0
        while done < height:
            _, bottom, _ = self.strips[self.next]
            n = min(height - done, bottom - self.end)
            at = self.end - self.start
            for name, rows in fields.items():
                if name not in self.buffers:
                    shape = (rows.shape[0], self.capacity, self.scene.width)
                    self.buffers[name] = np.empty(shape, dtype=rows.dtype)
                self.buffers[name][:, at : at + n] = rows[:, done : done + n]
            self.end += n
            done += n
            self._run_final()

    def _run_final(self) -> None:
        """Run every strip whose windows' rows are all in."""
        while self.next < len(self.strips) and self.end == self.strips[self.next][1]:
            for b in self.strips[self.next][2]:
                top, bottom, left, right = b.window
                inputs = self.post.given(
                    {
                        name: buffer[:, top - self.start : bottom - self.start, left:right].copy()
                        for name, buffer in self.buffers.items()
                    }
                )
                self.runner.submit(inputs, b)
            self.next += 1
            if self.next < len(self.strips):
                keep = self.strips[self.next][0]  # the next strip's windows start here
                kept = slice(keep - self.start, self.end - self.start)
                for buffer in self.buffers.values():
                    buffer[:, : self.end - keep] = buffer[:, kept]
                self.start = keep

    def close(self) -> Any:
        """Wait for the blocks; ``combine``'s result (or the list of results)."""
        if self.next < len(self.strips):
            raise AssertionError(f"only {self.end} of {self.scene.height} rows arrived")
        results = self.runner.finish()
        combine = self.post.combine
        return combine(results, self.scene) if combine is not None else results

    def abort(self) -> None:
        self.runner.abort()
