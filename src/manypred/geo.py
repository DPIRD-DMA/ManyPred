"""Raster reading and GeoTIFF writing (needs the ``[geo]`` extra: rasterio).

:func:`load_raster` and :func:`stack_rasters` read a scene as a ``(C, H, W)`` array plus the
rasterio profile of the grid it is on, optionally resampled to another resolution; they return
what a ``load_func`` for :func:`manypred.predict_files` returns. Which bands and files a model
needs (e.g. Sentinel-2 B04, B03, B8A) is the model's business, so there are no sensor-specific
loaders: a Sentinel-2 loader is a glob plus :func:`stack_rasters`.

The default ``save_func`` writes each dense output as a GeoTIFF on the grid described by a
rasterio profile in the loader's ``meta``, with the nodata mask as an internal mask band, and
image-level outputs to a JSON sidecar.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io_lock import IO_LOCK


def _rasterio():
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "Reading and writing GeoTIFFs needs rasterio: pip install 'manypred[geo]', or "
            "pass your own load_func and save_func to predict_files."
        ) from e
    return rasterio


def default_output_path(
    item: Any,
    output_dir: str | Path | None = None,
    *,
    suffix: str = "_pred",
    extension: str = ".tif",
) -> Path:
    """``<output_dir or the item's folder>/<item stem><suffix><extension>``. Creates
    ``output_dir``. A URI item (``s3://...``) needs a local ``output_dir``."""
    if not isinstance(item, (str, Path)) and not hasattr(item, "__fspath__"):
        raise TypeError(
            f"Items must be file paths to name outputs after them, got {type(item).__name__}. "
            f"Pass output_path=f(item, output_dir) -> path."
        )
    if output_dir is None and "://" in os.fspath(item):
        raise ValueError(
            f"{item} is a URI, so its output can't be written next to it. Pass output_dir, or "
            f"output_path=f(item, output_dir) -> path."
        )
    item = Path(item)
    name = f"{item.stem}{suffix}{extension}"
    if output_dir is None:
        return item.parent / name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(output_dir) / name


def head_path(path: str | Path, name: str) -> Path:
    """Where the default writer puts output ``name`` of a dict-output model whose item's
    output path is ``path``: ``<stem>_<name><suffix>``; ``path`` itself for ``name=""``."""
    path = Path(path)
    if not name:
        return path
    return path.with_name(f"{path.stem}_{name}{path.suffix}")


# profile keys of a source file's format and layout, which aren't the output's: an RGB file's
# photometric="ycbcr", for one, makes GDAL refuse any compression but JPEG
_LAYOUT = frozenset(
    {
        "driver",
        "blockxsize",
        "blockysize",
        "blocksize",
        "tiled",
        "compress",
        "interleave",
        "photometric",
        "jpeg_quality",
        "jpegtablesmode",
        "quality",
        "predictor",
        "zlevel",
        "zstd_level",
        "max_z_error",
        "nbits",
        "sparse_ok",
    }
)


def export_profile(
    profile: Mapping[str, Any] | None,
    array: np.ndarray,
    *,
    grid: tuple[int, int] | None = None,
    factor: int | None = None,
) -> dict[str, Any]:
    """A GeoTIFF profile for a (bands, H, W) array on ``profile``'s grid.

    - The profile's height, width, CRS and transform are kept if present. Loaders that
      resample can return the source grid's profile; GDAL then resizes the array as it writes.
    - The source's format and layout (driver, compression, blocks, interleave, photometric) are
      dropped, and nodata is unset: the mask band marks invalid pixels.
    - Floats are compressed with DEFLATE level 1 and the floating-point predictor (about half
      the size of LZW on logits, and faster to write); other dtypes with LZW.
    - ``BIGTIFF="IF_SAFER"`` unless the profile sets ``BIGTIFF``.
    - A reduced-resolution output, ceil(H / f) x ceil(W / f) for the (H, W) ``grid`` the model
      predicted on, when ``profile`` describes that grid, gets that size and pixels f times
      larger. ``factor`` is f; without it f is inferred from the sizes, which raises
      ValueError when more than one f fits (a 5 x 5 grid's 2 x 2 output is f = 3 or 4).
    """
    out = {key: value for key, value in (profile or {}).items() if key.lower() not in _LAYOUT}
    height, width = array.shape[1:]
    described = out.get("height"), out.get("width")
    if grid is not None and tuple(grid) == described and tuple(grid) != (height, width):
        f = _factor(tuple(grid), (height, width), factor)
        if f is not None:  # else the profile's grid stays, and GDAL resizes the array
            out.update(height=height, width=width)
            if out.get("transform") is not None:
                out["transform"] = _scaled(out["transform"], f, f)
    out.setdefault("height", height)
    out.setdefault("width", width)
    out.update(dtype=array.dtype, count=array.shape[0], nodata=None, driver="GTiff")
    # GDAL's default stays on classic TIFF when compressing, which fails partway at 4 GB
    if not any(key.upper() == "BIGTIFF" for key in out):
        out["BIGTIFF"] = "IF_SAFER"
    if np.issubdtype(array.dtype, np.floating):
        out.update(compress="deflate", zlevel=1, predictor=3)
    else:
        out.update(compress="lzw")
    return out


def _factor(grid: tuple[int, int], shape: tuple[int, int], factor: int | None) -> int | None:
    """The f for which ``shape`` is ceil(H / f) x ceil(W / f) of ``grid``: ``factor``, checked,
    or the only one that fits (None if none does). A 1 x 1 output (a head pooled over the
    whole grid) fits every f from max(H, W) up, and they all give one pixel over the whole
    grid: it gets the smallest, a pixel as large as the grid's longer side."""

    def fits(f: int) -> bool:
        return (-(-grid[0] // f), -(-grid[1] // f)) == tuple(shape)

    if factor is not None:
        if factor < 1 or not fits(factor):
            raise ValueError(
                f"A factor-{factor} output of a {grid[0]} x {grid[1]} grid is "
                f"ceil(H / {factor}) x ceil(W / {factor}), not {shape[0]} x {shape[1]}."
            )
        return factor
    fitting = [f for f in range(2, max(grid) + 2) if fits(f)]
    if len(fitting) > 1 and tuple(shape) != (1, 1):
        raise ValueError(
            f"A {shape[0]} x {shape[1]} output of a {grid[0]} x {grid[1]} grid could be "
            f"reduced by any of {fitting}; pass factor= to say which (or declare the "
            f"output's scale in the model's tiling spec)."
        )
    return fitting[0] if fitting else None


def _scaled(transform: Any, sx: float, sy: float) -> Any:
    """``transform`` with pixels ``sx`` x ``sy`` times larger, the origin kept (``transform *
    Affine.scale(sx, sy)``, without the operator newer affine versions deprecate)."""
    from affine import Affine

    a, b, c, d, e, f = tuple(transform)[:6]
    return Affine(a * sx, b * sy, c, d * sx, e * sy, f)


def _label(dst: Any, classes: Sequence[str] | None, input_bands: Any) -> None:
    """Names on an open GeoTIFF: class names as band descriptions when there is one band per
    class, else as a ``classes`` tag on a single integer band of class indices, and the input
    band names as an ``input_bands`` tag. Other shapes aren't per class, so they get no names."""
    if classes is not None:
        if dst.count == len(classes):
            dst.descriptions = tuple(classes)
        elif dst.count == 1 and np.issubdtype(np.dtype(dst.dtypes[0]), np.integer):
            dst.update_tags(1, classes=json.dumps(list(classes)))
    if input_bands is not None:
        dst.update_tags(input_bands=json.dumps(input_bands))


def _as_bands(array: Any) -> np.ndarray:
    """A dense output as a (bands, H, W) array GDAL can write: (H, W) gets one band, and bool
    becomes uint8."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"A GeoTIFF needs a (bands, H, W) or (H, W) array, got {array.shape}.")
    if array.dtype == bool:
        array = array.astype(np.uint8)
    return array


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    profile: Mapping[str, Any] | None = None,
    *,
    valid: np.ndarray | None = None,
    grid: tuple[int, int] | None = None,
    factor: int | None = None,
    classes: Sequence[str] | None = None,
    input_bands: Any = None,
) -> Path:
    """Write a (bands, H, W) or (H, W) array as a GeoTIFF (bool as uint8), via a temporary
    ``.<name>.partial`` file renamed when complete. ``valid`` (the (H, W) mask of the grid
    predicted on, ``grid``) becomes the mask band, pooled to the array's grid; ``factor`` is
    the array's reduction from ``grid`` (see :func:`export_profile`). ``classes`` and
    ``input_bands`` name what the file holds: class names as band descriptions (one band per
    class) or a ``classes`` tag (one band of class indices), input band names as a tag."""
    rio = _rasterio()
    path = Path(path)
    array = _as_bands(array)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Can't write {path}: {path.parent} is not a directory.")
    if grid is None and valid is not None:
        grid = tuple(np.shape(valid))
    if factor is None and grid is not None and tuple(grid) != array.shape[1:]:
        factor = _factor(tuple(grid), array.shape[1:], None)
    partial = _partial(path)
    try:
        # on one thread on purpose: with NUM_THREADS, GDAL can report success with blocks lost when
        # a compression thread runs out of memory, and reads freed memory if a mask follows the
        # bands. Saves that take their time only hold up the next load
        with (
            IO_LOCK.write(),
            rio.Env(GDAL_TIFF_INTERNAL_MASK=True),  # a sidecar would stay at the partial name
            rio.open(
                partial, "w", **export_profile(profile, array, grid=grid, factor=factor)
            ) as dst,
        ):
            dst.write(array)
            _label(dst, classes, input_bands)
            if valid is not None:
                mask = _pool_valid(np.asarray(valid), array.shape[1:], factor)
                dst.write_mask((mask * 255).astype(np.uint8))
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def _partial(path: Path) -> Path:
    """Where ``path`` is written until it is complete: a failed or killed write must not leave a
    file that a rerun with ``overwrite=False`` takes for a finished output. A killed run leaves
    this hidden file behind; the item's next write replaces it."""
    return path.with_name(f".{path.name}.partial")


def _remove_done_marks(*paths: Path) -> None:
    """Delete an item's earlier outputs that mark it done, before it is written again."""
    for path in paths:
        path.unlink(missing_ok=True)


def _stage_json(path: Path, value: Any) -> Path:
    """``value`` written as JSON to ``path``'s partial name, ready for :func:`_commit`."""
    partial = _partial(path)
    try:
        partial.write_text(json.dumps(value, indent=1))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial


def _write_json(path: Path, value: Any) -> None:
    _commit(_stage_json(path, value), path)


def _commit(
    manifest: Path, sidecar: Path, head: Path | None = None, base: Path | None = None
) -> None:
    """Move a staged manifest to ``sidecar``, and first, a staged "" head to ``base``: all the
    work is done by then, so the two renames follow each other and an item is marked done only
    when complete. If the manifest's rename fails, the head's is undone; only a process killed
    between the two renames can leave the head without its manifest."""
    try:
        if head is not None:
            os.replace(head, base)
        try:
            os.replace(manifest, sidecar)
        except BaseException:
            if head is not None:
                base.unlink(missing_ok=True)  # a lone base file would mark the item done
            raise
    finally:
        manifest.unlink(missing_ok=True)


def _pool_valid(valid: np.ndarray, shape: tuple[int, int], factor: int | None = None) -> np.ndarray:
    """A (H, W) valid mask on a ceil(H / f) x ceil(W / f) grid: valid where any pixel of the
    f x f block is (as for the arrays). ``factor`` is f, else inferred from the sizes."""
    if valid.shape == tuple(shape):
        return valid
    f = _factor(valid.shape, tuple(shape), factor)
    if f is None:
        raise ValueError(
            f"A mask of {valid.shape[0]} x {valid.shape[1]} doesn't reduce to "
            f"{shape[0]} x {shape[1]} by one integer factor."
        )
    padded = np.zeros((shape[0] * f, shape[1] * f), dtype=bool)
    padded[: valid.shape[0], : valid.shape[1]] = valid
    return padded.reshape(shape[0], f, shape[1], f).any(axis=(1, 3))


def _of_output(values: Mapping[str, Any] | None, name: str) -> Any:
    """An output's entry in ``{output: value}``; a single-output model's output has no name of
    its own, so it takes the only entry."""
    if not values:
        return None
    if not name and len(values) == 1:
        return next(iter(values.values()))
    return values.get(name)


class GeoTIFFWriter:
    """Default ``save_func``: GeoTIFFs named by ``output_path`` (``<stem>_pred.tif``).

    A single-output model's dense output goes to ``output_path(item, output_dir)``, and an
    image-level output (a 1D array, or :class:`~manypred.TilePredictions` reduced by
    ``.reduce("mean")``) to the same name with ``.json``. For a dict-output model each dense
    output goes to ``<stem>_pred_<name>.tif`` (:func:`head_path`) and a manifest
    ``<stem>_pred.json``, written last, lists those files and holds the image-level values;
    its presence marks the item done. ``meta`` should be a rasterio profile (or None for an
    ungeoreferenced file). Returns the GeoTIFF path for a single dense output, else the JSON
    path. Files are written under a hidden ``.<name>.partial`` name and renamed when complete.

    ``classes`` (``{output: names}``) and ``input_bands`` go into the GeoTIFFs and the JSON;
    :func:`manypred.predict_files` fills them from the model's ``tiling_spec()``.

    It is a block writer (``writes_blocks = True``): :func:`manypred.predict_files` calls
    :meth:`open` for lazy scenes instead of ``__call__``. A subclass that overrides
    ``__call__`` isn't one unless it sets ``writes_blocks = True`` itself."""

    writes_blocks = True
    uses_gdal = True  # predict_files keeps its block writes taking turns, whatever io_lock says

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # a subclass that saves differently would be bypassed by open() for lazy scenes
        if "__call__" in cls.__dict__ and "writes_blocks" not in cls.__dict__:
            cls.writes_blocks = False

    def __init__(self, output_path=default_output_path, classes=None, input_bands=None):
        self.output_path = output_path
        self.classes = classes or {}
        self.input_bands = input_bands
        # set by predict_files: check_heads(names) refuses head files that would replace an
        # input or another item's output
        self.check_heads: Callable[[list[str]], None] | None = None

    def open(
        self,
        meta: Any,
        item: Any,
        output_dir: Any,
        grid: tuple[int, int] | None = None,
    ) -> GeoTIFFBlocks:
        """A block writer for one item: :func:`manypred.predict_files` uses it for lazy
        scenes, writing rows as they are finished instead of holding the whole output."""
        return GeoTIFFBlocks(
            Path(self.output_path(item, output_dir)),
            meta,
            grid,
            self.classes,
            self.input_bands,
            check_heads=self.check_heads,
        )

    def existing(self, item: Any, output_dir: Any) -> Path | None:
        """The output of an item already written, if any (for ``overwrite=False``)."""
        path = Path(self.output_path(item, output_dir))
        for candidate in (path, path.with_suffix(".json")):
            if candidate.exists():
                return candidate
        return None

    def __call__(
        self,
        prediction: Any,
        meta: Any,
        item: Any,
        output_dir: Any,
        valid: np.ndarray | None = None,
        grid: tuple[int, int] | None = None,
        factors: Mapping[str, int | None] | None = None,
    ) -> Path:
        path = Path(self.output_path(item, output_dir))
        if not isinstance(prediction, Mapping):
            value = _as_array(prediction)
            if value.ndim >= 2:
                # ``path`` marks this item done once written; an earlier manifest must not
                _remove_done_marks(path.with_suffix(".json"))
                return write_geotiff(
                    path,
                    value,
                    meta,
                    valid=valid,
                    grid=grid,
                    factor=_of_output(factors, ""),
                    classes=_of_output(self.classes, ""),
                    input_bands=self.input_bands,
                )
            sidecar = path.with_suffix(".json")
            _remove_done_marks(path)  # an earlier dense output's must not mark it done
            single = {"prediction": value.tolist()}
            _write_json(sidecar, _named(single, _of_output(self.classes, ""), self.input_bands))
            return sidecar
        values = {name: _as_array(value) for name, value in prediction.items()}
        dense = [name for name, value in values.items() if value.ndim >= 2]
        if self.check_heads is not None:
            self.check_heads(dense)  # before any file is written
        sidecar = path.with_suffix(".json")
        # the manifest, and ``path`` (a "" head's file), each mark the item done (``existing``):
        # an earlier run's must not outlive a failure partway through replacing the heads
        _remove_done_marks(sidecar, path)
        staged = path.with_name(f".{path.name}.staged")  # a "" head, until the manifest is
        files = {}
        try:
            for name in dense:
                write_geotiff(
                    staged if name == "" else head_path(path, name),
                    values[name],
                    meta,
                    valid=valid,
                    grid=grid,
                    factor=(factors or {}).get(name),
                    classes=self.classes.get(name),
                    input_bands=self.input_bands,
                )
                files[name] = head_path(path, name).name
            manifest = {
                "files": files,
                "values": {n: v.tolist() for n, v in values.items() if n not in files},
            }
            _commit(
                _stage_json(sidecar, _named(manifest, self.classes, self.input_bands)),
                sidecar,
                staged if "" in files else None,
                path,
            )
        finally:
            staged.unlink(missing_ok=True)
        return sidecar


def _named(content: dict[str, Any], classes: Any, input_bands: Any) -> dict[str, Any]:
    """A JSON's content with the names that apply. No names, no keys."""
    if classes:
        content["classes"] = classes
    if input_bands is not None:
        content["input_bands"] = input_bands
    return content


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "reduce"):  # TilePredictions
        value = value.reduce("mean")
    return np.asarray(value)


class GeoTIFFBlocks:
    """Writes outputs to GeoTIFFs block by block (same files as :class:`GeoTIFFWriter`).
    Each output's file is created when its first block arrives."""

    # GDAL's block cache defaults to 5% of RAM and holds written blocks until it fills, so
    # without a cap memory grows with the output size
    CACHE_BYTES = 64 * 2**20

    def __init__(
        self,
        path: Path,
        profile: Mapping[str, Any] | None,
        grid: tuple[int, int] | None = None,
        classes: Mapping[str, Any] | None = None,
        input_bands: Any = None,
        *,
        check_heads: Callable[[list[str]], None] | None = None,
    ):
        _rasterio()  # fail here rather than at the first block
        self.path = path
        self.profile = profile
        self.grid = grid
        self.classes = classes or {}
        self.input_bands = input_bands
        self.check_heads = check_heads
        self.datasets: dict[str, Any] = {}
        self.paths: dict[str, Path] = {}
        self.env: Any = None  # entered with the first file, so nothing is held before then

    def write(
        self,
        name: str,
        top: int,
        block: np.ndarray,
        shape: tuple[int, int, int],
        valid: np.ndarray | None,
        factor: int | None = None,
    ) -> None:
        """Rows [top, top + h) of output ``name`` (full shape ``shape``, ``factor`` times
        coarser than the grid predicted on), and their mask."""
        rio = _rasterio()
        from rasterio.windows import Window

        if block.dtype == bool:
            block = block.astype(np.uint8)
        if name not in self.datasets:
            if name and self.check_heads is not None:
                self.check_heads([name])
            if self.env is None:
                self.env = rio.Env(GDAL_CACHEMAX=self.CACHE_BYTES, GDAL_TIFF_INTERNAL_MASK=True)
                self.env.__enter__()
            # a zero-memory stand-in for the whole output, to build its profile
            template = np.broadcast_to(np.zeros((), dtype=block.dtype), shape)
            self.paths[name] = head_path(self.path, name)
            # written in rows, whatever the source's layout (export_profile drops it): rows
            # written into tiles make the file larger and slower to write
            self.datasets[name] = rio.open(
                _partial(self.paths[name]),
                "w",
                **export_profile(self.profile, template, grid=self.grid, factor=factor),
            )
            _label(self.datasets[name], _of_output(self.classes, name), self.input_bands)
        window = Window(0, top, block.shape[2], block.shape[1])
        dst = self.datasets[name]
        dst.write(block, window=window)
        if valid is not None:
            dst.write_mask((valid * 255).astype(np.uint8), window=window)

    def _close_files(self) -> None:
        try:
            for dataset in self.datasets.values():
                dataset.close()
        finally:
            self.datasets = {}
            if self.env is not None:
                self.env.__exit__(None, None, None)
                self.env = None

    def abort(self) -> None:
        """After an error: close and delete the partial files, so a rerun with
        ``overwrite=False`` does not take them for finished outputs."""
        self._close_files()
        for path in self.paths.values():
            _partial(path).unlink(missing_ok=True)

    def close(self, extras: Mapping[str, Any] | None = None) -> Path:
        """Close the files and move them to their names; image-level outputs (``extras``) go to
        the JSON as with :class:`GeoTIFFWriter`. Returns the same path it would."""
        self._close_files()
        extras = dict(extras or {})
        names = set(self.paths) | set(extras)
        sidecar = self.path.with_suffix(".json")
        if names == {""} and "" in self.paths:  # one dense output, at ``self.path``
            _remove_done_marks(sidecar)  # an earlier manifest must not mark the item done
            os.replace(_partial(self.paths[""]), self.paths[""])
            return self.paths[""]
        # the manifest, and ``self.path`` (a "" head's file), each mark the item done: an
        # earlier run's must not outlive a failure partway through
        _remove_done_marks(sidecar, self.path)
        for name, path in self.paths.items():
            if name:
                os.replace(_partial(path), path)
        manifest: dict[str, Any] = {
            "files": {name: path.name for name, path in self.paths.items()},
            "values": {name: _as_array(value).tolist() for name, value in extras.items()},
        }
        content = {"prediction": manifest["values"][""]} if names == {""} else manifest
        classes = _of_output(self.classes, "") if names == {""} else self.classes
        head = _partial(self.paths[""]) if "" in self.paths else None
        _commit(
            _stage_json(sidecar, _named(content, classes, self.input_bands)),
            sidecar,
            head,
            self.path,
        )
        return sidecar


def _resampling(name: str) -> Any:
    from rasterio.enums import Resampling

    try:
        return Resampling[name]
    except KeyError:
        raise ValueError(
            f"Unknown resampling {name!r}; use one of {[r.name for r in Resampling]}."
        ) from None


def _grid(src: Any, resolution: float | None) -> tuple[int, int, Any]:
    """(height, width, transform) of ``src``'s extent with pixels of ``resolution`` (in CRS
    units), the extent kept exactly (so pixels differ from ``resolution`` by under a pixel's
    worth over the extent when it doesn't divide), or the native grid for None."""
    if resolution is None:
        return src.height, src.width, src.transform
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")
    t = src.transform
    if t.b or t.d:
        raise ValueError(
            "resolution= needs a grid without rotation; reproject the file first "
            "(rasterio.warp.reproject)."
        )
    xres, yres = src.res
    height = max(1, round(src.height * yres / resolution))
    width = max(1, round(src.width * xres / resolution))
    return height, width, _scaled(src.transform, src.width / width, src.height / height)


def _read(src: Any, bands: list[int], height: int, width: int, resampling: str) -> np.ndarray:
    if (height, width) == (src.height, src.width):
        return src.read(bands)
    return src.read(
        bands, out_shape=(len(bands), height, width), resampling=_resampling(resampling)
    )


def _check_grid(grid: str) -> None:
    if grid not in ("array", "source"):
        raise ValueError(f"grid must be 'array' or 'source', got {grid!r}.")


def _profile(src: Any, data: np.ndarray, transform: Any, grid: str = "array") -> dict[str, Any]:
    """The profile of ``data`` (``grid="array"``: its own grid) or of ``src``'s grid
    (``"source"``), with ``data``'s band count and dtype."""
    profile = dict(src.profile)
    profile.update(count=data.shape[0], dtype=str(data.dtype))
    if grid == "array":
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    return _without_layout(profile)


def _without_layout(profile: Mapping[str, Any]) -> dict[str, Any]:
    """A profile without the source file's format and layout, which aren't the output's."""
    return {key: value for key, value in profile.items() if key.lower() not in _LAYOUT}


def _bands(bands: list[int] | None, count: int, path: Any) -> list[int]:
    """``bands`` (1-based) checked against a file's ``count``; None means all."""
    if bands is None:
        return list(range(1, count + 1))
    bands = list(bands)
    if not bands:
        raise ValueError("bands=[] selects no bands.")
    bad = [b for b in bands if not isinstance(b, (int, np.integer)) or not 1 <= b <= count]
    if bad:
        raise ValueError(f"{path} has bands 1-{count}; asked for {bad}.")
    return bands


@IO_LOCK.read()
def load_raster(
    path: str | Path,
    *,
    bands: list[int] | None = None,
    resolution: float | None = None,
    resampling: str = "bilinear",
    grid: str = "array",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster as ``(array, profile)``: ``array`` is (C, H, W) in the file's dtype,
    ``profile`` describes its grid (resampled or not), ready for the default ``save_func``. The
    profile keeps the file's CRS and nodata value but not its format (driver, compression,
    blocks, photometric). ``predict_files`` takes nodata from its own ``nodata=`` (default
    None), not from the profile.

    Args:
        path: The raster file.
        bands: 1-based band numbers, in the order wanted; default all.
        resolution: Pixel size in CRS units (e.g. 10 for 10 m) to resample to, keeping the
            extent; default the file's own.
        resampling: A rasterio resampling name: ``"bilinear"`` (as OmniCloudMask),
            ``"average"`` (better when making pixels coarser), ``"nearest"`` (classes).
        grid: Which grid the returned profile describes. ``"array"`` (the default): the
            returned (resampled) grid, so predictions are written at ``resolution``.
            ``"source"``: the file's own grid, so the default writer writes predictions back
            at the file's resolution (GDAL resizes them with nearest neighbour as it writes).
    """
    rio = _rasterio()
    with rio.open(path) as src:
        _resampling(resampling)
        _check_grid(grid)
        bands = _bands(bands, src.count, path)
        height, width, transform = _grid(src, resolution)
        data = _read(src, bands, height, width, resampling)
        return data, _profile(src, data, transform, grid)


@IO_LOCK.read()
def stack_rasters(
    paths: list[str | Path],
    *,
    resolution: float | None = None,
    resampling: str = "bilinear",
    grid: str = "array",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stack the bands of several rasters covering the same extent (e.g. one file per band, at
    10, 20 and 60 m) into one ``(array, profile)`` at one resolution.

    Args:
        paths: Files, in the order their bands are wanted; every band of each is used.
        resolution: Pixel size in CRS units; default the finest of the files.
        resampling: As for :func:`load_raster`.
        grid: As for :func:`load_raster`; ``"source"`` is the first file's grid.

    The files must share a CRS and extent (to 1% of a pixel) on north-up grids: each is resized
    over its own extent, not reprojected. The profile's nodata is the files' if they agree,
    else None.
    """
    rio = _rasterio()
    paths = list(paths)
    if not paths:
        raise ValueError("stack_rasters needs at least one file.")
    _resampling(resampling)
    _check_grid(grid)
    if resolution is None:
        resolutions = []
        for path in paths:
            with rio.open(path) as src:
                resolutions.append(min(src.res))
        resolution = min(resolutions)
    arrays, out, nodata = [], None, set()
    for path in paths:
        with rio.open(path) as src:
            t = src.transform
            if t.b or t.d or t.a <= 0 or t.e >= 0:
                raise ValueError(
                    f"{path}: stack_rasters needs north-up grids without rotation; reproject "
                    f"the file first (rasterio.warp.reproject)."
                )
            if out is None:
                height, width, transform = _grid(src, resolution)
                crs, bounds = src.crs, src.bounds
            elif src.crs != crs:
                raise ValueError(f"{path} is in {src.crs}, {paths[0]} in {crs}.")
            elif any(
                abs(a - b) > resolution / 100 for a, b in zip(src.bounds, bounds, strict=True)
            ):
                # each file is resized over its own extent, so the extents must agree for the
                # pixels to line up (to 1% of a pixel)
                raise ValueError(
                    f"{path} covers {tuple(src.bounds)}, {paths[0]} {tuple(bounds)}: "
                    f"stack_rasters needs files on the same extent; reproject the others onto "
                    f"one grid first (rasterio.warp.reproject)."
                )
            nodata.add(src.nodata)
            arrays.append(_read(src, list(range(1, src.count + 1)), height, width, resampling))
            if out is None:
                out = _profile(src, arrays[0], transform, grid)
    data = np.concatenate(arrays)
    # one nodata value if the files agree, else none (their bands mean different things by it)
    out.update(
        count=data.shape[0],
        dtype=str(data.dtype),
        nodata=nodata.pop() if len(nodata) == 1 else None,
    )
    return data, out


class RasterioReader:
    """A lazy reader over a raster file for :func:`manypred.predict_files`: ``shape``,
    ``dtype``, ``profile`` and ``read((top, bottom, left, right))``. Opens the file per read,
    so it is safe to use from the loader thread; ``bands`` selects bands (1-based), which must
    share a dtype. ``profile`` keeps the file's grid, CRS and nodata value but not its format
    (driver, compression, blocks, photometric), as :func:`load_raster`'s does, and
    ``predict_files`` uses it when ``load_func`` gives no ``meta``."""

    def __init__(self, path: str | Path, bands: list[int] | None = None):
        rio = _rasterio()
        self.path = Path(path)
        with IO_LOCK.read(), rio.open(self.path) as src:
            self.bands = _bands(bands, src.count, self.path)
            dtypes = {src.dtypes[b - 1] for b in self.bands}
            if len(dtypes) > 1:
                raise ValueError(
                    f"{self.path}: bands {self.bands} have different dtypes {sorted(dtypes)}, "
                    f"which rasterio can't read together; pass bands= of one dtype."
                )
            self.shape = (len(self.bands), src.height, src.width)
            self.dtype = np.dtype(dtypes.pop())
            self.profile = _without_layout(src.profile)
            self.profile.update(count=len(self.bands), dtype=str(self.dtype))

    @IO_LOCK.read()
    def read(self, window: tuple[int, int, int, int]) -> np.ndarray:
        rio = _rasterio()
        from rasterio.windows import Window

        top, bottom, left, right = window
        with rio.open(self.path) as src:
            return src.read(self.bands, window=Window(left, top, right - left, bottom - top))
