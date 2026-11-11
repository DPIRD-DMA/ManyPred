"""The default writer and the geo helpers: profiles, factors, output names and guards."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import SeededConv

import manypred.files as files
import manypred.geo as geo
from manypred import predict_files
from manypred.geo import (
    GeoTIFFBlocks,
    GeoTIFFWriter,
    RasterioReader,
    default_output_path,
    export_profile,
    load_raster,
    stack_rasters,
    write_geotiff,
)
from manypred.lazy import ArrayReader, band_rows, is_reader

rio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

KW = {"progress": False, "device": "cpu", "nodata": None}
TRANSFORM = Affine(10, 0, 400000, 0, -10, 6500000)


def write_tif(path, array, **extra):
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": "EPSG:32750",
        "transform": TRANSFORM,
        **extra,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
    return Path(path)


def rgb_ycbcr(path, size=64):
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (3, size, size), dtype=np.uint8)
    extra = {"compress": "jpeg", "photometric": "ycbcr", "tiled": True}
    return write_tif(path, array, blockxsize=32, blockysize=32, **extra)


class TwoHeads(torch.nn.Module):
    """A full-resolution head and one 4 times coarser (ceil-sized)."""

    def forward(self, x):
        full = x[:, :1] * 2
        return {"full": full, "coarse": F.avg_pool2d(full, 4, ceil_mode=True)}


# --- profiles -------------------------------------------------------------------------------


def test_export_profile_drops_the_source_layout():
    profile = {
        "height": 4,
        "width": 4,
        "photometric": "ycbcr",
        "INTERLEAVE": "pixel",
        "tiled": True,
        "blockxsize": 512,
        "jpeg_quality": 90,
        "compress": "jpeg",
    }
    out = export_profile(profile, np.zeros((1, 4, 4), np.float32))
    assert not {"photometric", "INTERLEAVE", "tiled", "blockxsize", "jpeg_quality"} & set(out)
    assert out["compress"] == "deflate"


def test_writes_on_the_profile_of_a_ycbcr_jpeg_file(tmp_path):
    path = rgb_ycbcr(tmp_path / "rgb.tif")
    _, profile = load_raster(path)
    assert "photometric" not in profile
    out = write_geotiff(tmp_path / "out.tif", np.zeros((2, 64, 64), np.float32), profile)
    with rio.open(out) as src:
        assert src.count == 2 and src.crs == "EPSG:32750"
    # a raw profile, as a load_func returning src.profile gives
    with rio.open(path) as src:
        raw = src.profile
    write_geotiff(tmp_path / "raw.tif", np.zeros((1, 64, 64), np.uint8), raw)


def test_predict_files_on_a_ycbcr_jpeg_file_in_memory_and_lazily(tmp_path):
    path = rgb_ycbcr(tmp_path / "rgb.tif")
    in_memory = predict_files(
        SeededConv(),
        [path],
        load_func=load_raster,
        output_dir=tmp_path / "a",
        patch_size=32,
        overlap=8,
        **KW,
    )
    lazy = predict_files(
        SeededConv(),
        [path],
        load_func=RasterioReader,
        output_dir=tmp_path / "b",
        patch_size=32,
        overlap=8,
        **KW,
    )
    for out in (in_memory[0], lazy[0]):
        with rio.open(out) as src:
            assert src.shape == (64, 64)


# --- write_geotiff inputs -------------------------------------------------------------------


def test_write_geotiff_takes_2d_and_bool_arrays(tmp_path):
    out = write_geotiff(tmp_path / "a.tif", np.ones((8, 8), bool))
    with rio.open(out) as src:
        assert src.count == 1 and src.dtypes[0] == "uint8" and src.read(1).all()


def test_write_geotiff_refuses_other_shapes_and_missing_folders(tmp_path):
    with pytest.raises(ValueError, match=r"\(bands, H, W\)"):
        write_geotiff(tmp_path / "a.tif", np.zeros((1, 1, 8, 8), np.float32))
    with pytest.raises(FileNotFoundError, match="missing"):
        write_geotiff(tmp_path / "missing" / "a.tif", np.zeros((1, 8, 8), np.float32))
    assert not any((tmp_path).rglob("*.partial"))


def test_a_2d_dense_output_is_written_as_a_geotiff(tmp_path):
    item = write_tif(tmp_path / "a.tif", np.ones((3, 32, 32), np.float32))
    [out] = predict_files(
        SeededConv(),
        [item],
        load_func=load_raster,
        output=lambda x: x[0],
        output_dir=tmp_path / "out",
        patch_size=32,
        overlap=0,
        **KW,
    )
    assert out.suffix == ".tif" and not out.with_suffix(".json").exists()
    with rio.open(out) as src:
        assert src.shape == (32, 32) and src.count == 1


# --- reduced-resolution outputs -------------------------------------------------------------


def test_the_factor_given_sets_pixel_size_and_mask():
    profile = {"height": 5, "width": 5, "transform": TRANSFORM}
    array = np.zeros((1, 2, 2), np.float32)
    with pytest.raises(ValueError, match="pass factor"):
        export_profile(profile, array, grid=(5, 5))  # f = 3 or 4
    assert export_profile(profile, array, grid=(5, 5), factor=4)["transform"].a == 40
    with pytest.raises(ValueError, match="factor-2"):
        export_profile(profile, array, grid=(5, 5), factor=2)
    valid = np.zeros((5, 5), bool)
    valid[3, 3] = True  # the first 4 x 4 block; the second 3 x 3 one
    assert geo._pool_valid(valid, (2, 2), 4).tolist() == [[True, False], [False, False]]


def test_a_one_pixel_output_covers_the_grid():
    """Every f from max(H, W) up makes a 1 x 1 output of the whole grid; it was refused as
    ambiguous."""
    profile = {"height": 5, "width": 7, "transform": TRANSFORM}
    out = export_profile(profile, np.zeros((1, 1, 1), np.float32), grid=(5, 7))
    assert (out["height"], out["width"]) == (1, 1)
    assert out["transform"].a == 7 * TRANSFORM.a
    valid = np.zeros((5, 7), bool)
    valid[4, 6] = True
    assert geo._pool_valid(valid, (1, 1)).tolist() == [[True]]


def test_a_one_pixel_output_is_written_without_a_factor(tmp_path):
    valid = np.ones((5, 7), bool)
    path = write_geotiff(tmp_path / "one.tif", np.ones((1, 1, 1), np.float32), valid=valid)
    with rio.open(path) as src:
        assert (src.height, src.width) == (1, 1)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_coarse_head_gets_its_true_pixel_size(tmp_path, lazy):
    # a 9 x 9 scene's factor-4 head is 3 x 3, which factor 3 would give too
    item = write_tif(tmp_path / "s.tif", np.ones((1, 9, 9), np.float32))
    [manifest] = predict_files(
        TwoHeads(),
        [item],
        load_func=RasterioReader if lazy else load_raster,
        output_dir=tmp_path / "out",
        patch_size=8,
        overlap=0,
        **KW,
    )
    files_ = json.loads(manifest.read_text())["files"]
    with rio.open(manifest.parent / files_["coarse"]) as src:
        assert src.shape == (3, 3) and src.transform.a == 40 and src.transform.e == -40
    with rio.open(manifest.parent / files_["full"]) as src:
        assert src.shape == (9, 9) and src.transform.a == 10


def test_batch_mode_uses_the_spec_factor(tmp_path):
    items = [write_tif(tmp_path / f"c{i}.tif", np.ones((1, 5, 5), np.float32)) for i in range(2)]
    spec = {
        "outputs": {
            "full": {"scope": "pixel", "scale": 1.0, "kind": "regression"},
            "coarse": {"scope": "pixel", "scale": 0.25, "kind": "regression"},
        }
    }
    out = predict_files(
        TwoHeads(),
        items,
        load_func=load_raster,
        mode="batch",
        spec=spec,
        output_dir=tmp_path / "out",
        **KW,
    )
    for manifest in out:
        name = json.loads(manifest.read_text())["files"]["coarse"]
        with rio.open(manifest.parent / name) as src:
            assert src.shape == (2, 2) and src.transform.a == 40


class Halved(torch.nn.Module):
    def forward(self, x):
        return {"seg": F.avg_pool2d(x[:, :1], 2)}


def test_batch_mode_infers_factors_the_spec_does_not_declare(tmp_path):
    items = [write_tif(tmp_path / f"c{i}.tif", np.ones((1, 8, 8), np.float32)) for i in range(2)]
    out = predict_files(
        Halved(),
        items,
        load_func=load_raster,
        mode="batch",
        spec={"outputs": {"seg": {"activation": "none"}}},  # no scale
        output_dir=tmp_path / "out",
        **KW,
    )
    for manifest in out:
        name = json.loads(manifest.read_text())["files"]["seg"]
        with rio.open(manifest.parent / name) as src:
            assert src.shape == (4, 4) and src.transform.a == 20


# --- output names ---------------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
def test_a_head_file_never_replaces_an_input(tmp_path, lazy):
    a = write_tif(tmp_path / "a.tif", np.ones((1, 16, 16), np.float32))
    b = write_tif(tmp_path / "a_pred_full.tif", np.full((1, 16, 16), 3, np.float32))
    before = b.read_bytes()
    with pytest.raises(ValueError, match="also an input"):
        predict_files(
            TwoHeads(),
            [a, b],
            load_func=RasterioReader if lazy else load_raster,
            patch_size=16,
            overlap=0,
            **KW,
        )
    assert b.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tif", "a_pred_full.tif"]


def test_uri_items_need_an_output_dir(tmp_path):
    with pytest.raises(ValueError, match="output_dir"):
        default_output_path("s3://bucket/dir/a.tif")
    assert default_output_path("s3://bucket/dir/a.tif", tmp_path) == tmp_path / "a_pred.tif"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS case handling")
def test_case_only_names_differ_on_a_case_sensitive_volume(tmp_path, monkeypatch):
    assert files._path_key(tmp_path / "A.tif") == files._path_key(tmp_path / "a.tif")
    monkeypatch.setattr(files, "_ignores_case", lambda directory: False)
    assert files._path_key(tmp_path / "A.tif") != files._path_key(tmp_path / "a.tif")


# --- block writers --------------------------------------------------------------------------


def test_a_subclass_that_saves_differently_is_not_a_block_writer(tmp_path):
    calls = []

    class Custom(GeoTIFFWriter):
        def __call__(self, prediction, meta, item, output_dir, **kwargs):
            calls.append(item)
            return super().__call__(prediction, meta, item, output_dir, **kwargs)

    class Streams(Custom):
        writes_blocks = True

    assert GeoTIFFWriter.writes_blocks and not Custom.writes_blocks and Streams.writes_blocks
    item = write_tif(tmp_path / "a.tif", np.ones((3, 32, 32), np.float32))
    with pytest.warns(UserWarning, match="held in memory"):
        predict_files(
            SeededConv(),
            [item],
            load_func=RasterioReader,
            save_func=Custom(),
            patch_size=32,
            overlap=0,
            **KW,
        )
    assert calls == [item]


def test_open_alone_does_not_make_a_block_writer(tmp_path):
    class Saver:
        def __init__(self):
            self.calls = 0

        def __call__(self, prediction, meta, item, output_dir):
            self.calls += 1
            return item

        def open(self, *args, **kwargs):
            raise AssertionError("not a block writer")

    item = write_tif(tmp_path / "a.tif", np.ones((3, 32, 32), np.float32))
    saver = Saver()
    with pytest.warns(UserWarning, match="held in memory"):
        predict_files(
            SeededConv(),
            [item],
            load_func=RasterioReader,
            save_func=saver,
            patch_size=32,
            overlap=0,
            **KW,
        )
    assert saver.calls == 1


def test_block_writer_enters_gdal_settings_only_while_writing(tmp_path):
    writer = GeoTIFFBlocks(tmp_path / "a.tif", None, (8, 8))
    assert writer.env is None  # nothing to leak if the run fails before the first block
    writer.write("", 0, np.zeros((1, 8, 8), np.float32), (1, 8, 8), None, factor=1)
    assert writer.env is not None
    writer.abort()
    assert writer.env is None and not any(tmp_path.iterdir())


def test_a_failure_before_the_first_block_aborts_the_writer(tmp_path, monkeypatch):
    from manypred.blocks import BlockPostprocess

    aborted = []
    original = GeoTIFFBlocks.abort

    def abort(self):
        aborted.append(True)
        original(self)

    def broken(self, scene):
        raise RuntimeError("no stream")

    monkeypatch.setattr(GeoTIFFBlocks, "abort", abort)
    monkeypatch.setattr(BlockPostprocess, "stream", broken)
    item = write_tif(tmp_path / "a.tif", np.ones((3, 32, 32), np.float32))
    with pytest.raises(RuntimeError, match="no stream"):
        predict_files(
            SeededConv(),
            [item],
            load_func=RasterioReader,
            block_postprocess=BlockPostprocess(lambda fields, block, scene: None),
            patch_size=32,
            overlap=0,
            **KW,
        )
    assert aborted


# --- rasterio missing -----------------------------------------------------------------------


def test_missing_rasterio_fails_before_any_load(tmp_path, monkeypatch):
    def missing():
        raise ImportError("Reading and writing GeoTIFFs needs rasterio")

    monkeypatch.setattr(geo, "_rasterio", missing)
    loads = []
    with pytest.raises(ImportError, match="needs rasterio"):
        predict_files(
            SeededConv(),
            [tmp_path / "a.tif"],
            load_func=lambda item: loads.append(item) or np.ones((3, 8, 8), np.float32),
            **KW,
        )
    assert not loads


# --- loaders and readers --------------------------------------------------------------------


def test_load_raster_and_stack_rasters_take_grid(tmp_path):
    path = write_tif(tmp_path / "a.tif", np.ones((1, 8, 8), np.uint16))
    _, profile = load_raster(path, resolution=20, grid="source")
    assert profile["height"] == 8
    _, profile = stack_rasters([path], resolution=20, grid="array")
    assert profile["height"] == 4
    with pytest.raises(ValueError, match="grid must be"):
        load_raster(path, grid="other")
    with pytest.raises(TypeError):
        load_raster(path, profile="source")


def test_rasterio_reader_refuses_bands_of_mixed_dtypes(tmp_path):
    write_tif(tmp_path / "u8.tif", np.ones((1, 4, 4), np.uint8))
    write_tif(tmp_path / "f4.tif", np.ones((1, 4, 4), np.float32))
    bands = "".join(
        f'<VRTRasterBand dataType="{dtype}" band="{i}"><SimpleSource>'
        f'<SourceFilename relativeToVRT="1">{name}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource></VRTRasterBand>"
        for i, (dtype, name) in enumerate([("Byte", "u8.tif"), ("Float32", "f4.tif")], 1)
    )
    vrt = tmp_path / "mixed.vrt"
    vrt.write_text(f'<VRTDataset rasterXSize="4" rasterYSize="4">{bands}</VRTDataset>')
    with pytest.raises(ValueError, match="different dtypes"):
        RasterioReader(vrt)
    assert RasterioReader(vrt, bands=[2]).dtype == np.float32


def test_band_rows_counts_every_frame_of_a_time_series():
    frames = ArrayReader(np.zeros((4, 2, 10, 100), np.float32))
    single = ArrayReader(np.zeros((2, 10, 100), np.float32))
    budget = 4 * 2 * 100 * 4 * 5  # five rows of every frame
    assert band_rows(frames, budget) == 5
    assert band_rows(single, budget) == 20


def test_an_open_rasterio_dataset_is_refused_with_a_hint(tmp_path):
    path = write_tif(tmp_path / "a.tif", np.ones((1, 4, 4), np.uint8))
    with rio.open(path) as src, pytest.raises(TypeError, match="RasterioReader"):
        is_reader(src)
    assert is_reader(ArrayReader(np.zeros((1, 4, 4))))
    assert not is_reader(np.zeros((4, 4)))


@pytest.mark.parametrize("heads", [("a", "b"), ("", "b")])
def test_a_failed_overwrite_of_heads_leaves_the_item_unfinished(tmp_path, monkeypatch, heads):
    """An earlier run's manifest (or base file, for a "" head) must not mark an item done once
    a later run has started replacing its heads and failed."""
    writer = GeoTIFFWriter()
    item = tmp_path / "scene.tif"
    prediction = {name: np.zeros((1, 4, 4), dtype=np.float32) for name in heads}
    writer(prediction, None, item, tmp_path)
    assert writer.existing(item, tmp_path) is not None

    real = geo.write_geotiff
    written = []

    def fail_second(path, *args, **kwargs):
        written.append(path)
        if len(written) == 2:
            raise OSError("disk full")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(geo, "write_geotiff", fail_second)
    with pytest.raises(OSError):
        writer(prediction, None, item, tmp_path)
    assert writer.existing(item, tmp_path) is None
    assert not default_output_path(item, tmp_path).exists()


def test_a_failed_manifest_leaves_a_bare_head_unfinished(tmp_path, monkeypatch):
    writer = GeoTIFFWriter()
    item = tmp_path / "scene.tif"
    prediction = {"": np.zeros((1, 4, 4), np.float32), "score": np.array([0.5])}

    def broken(path, value):
        raise OSError("disk full")

    monkeypatch.setattr(geo, "_stage_json", broken)
    with pytest.raises(OSError):
        writer(prediction, None, item, tmp_path)
    assert writer.existing(item, tmp_path) is None
    assert not list(tmp_path.glob(".*"))  # no staged or partial files left


def test_an_old_bare_head_does_not_outlive_a_failed_rewrite(tmp_path, monkeypatch):
    writer = GeoTIFFWriter()
    item = tmp_path / "scene.tif"
    zeros = np.zeros((1, 4, 4), np.float32)
    writer({"": zeros, "b": zeros}, None, item, tmp_path)
    real = geo.write_geotiff
    written = []

    def fail_second(path, *args, **kwargs):
        written.append(path)
        if len(written) == 2:
            raise OSError("disk full")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(geo, "write_geotiff", fail_second)
    with pytest.raises(OSError):
        writer({"a": zeros, "b": zeros}, None, item, tmp_path)
    assert writer.existing(item, tmp_path) is None


def test_a_dict_output_with_a_bare_head_round_trips(tmp_path):
    writer = GeoTIFFWriter()
    item = tmp_path / "scene.tif"
    zeros = np.zeros((1, 4, 4), np.float32)
    out = writer({"": zeros, "b": zeros, "score": np.array([0.5])}, None, item, tmp_path)
    manifest = json.loads(out.read_text())
    assert manifest["files"] == {"": "scene_pred.tif", "b": "scene_pred_b.tif"}
    assert manifest["values"] == {"score": [0.5]}
    assert writer.existing(item, tmp_path) is not None
    assert not list(tmp_path.glob(".*"))


def test_a_failed_block_close_leaves_no_old_manifest(tmp_path):
    item = tmp_path / "scene.tif"
    writer = GeoTIFFWriter()
    writer({"a": np.zeros((1, 4, 4), np.float32)}, None, item, tmp_path)
    sidecar = default_output_path(item, tmp_path).with_suffix(".json")
    assert sidecar.exists()
    blocks = writer.open(None, item, tmp_path, grid=(4, 4))
    blocks.write("a", 0, np.ones((1, 4, 4), np.float32), (1, 4, 4), None)
    blocks.paths["b"] = blocks.path.with_name("missing_b.tif")  # its partial never written
    with pytest.raises(FileNotFoundError):
        blocks.close()
    assert not sidecar.exists()


def test_a_failed_manifest_rename_undoes_the_bare_head(tmp_path, monkeypatch):
    writer = GeoTIFFWriter()
    item = tmp_path / "scene.tif"
    zeros = np.zeros((1, 4, 4), np.float32)
    real = geo.os.replace

    def replace(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(geo.os, "replace", replace)
    with pytest.raises(OSError):
        writer({"": zeros, "b": zeros}, None, item, tmp_path)
    assert writer.existing(item, tmp_path) is None


@pytest.mark.parametrize("group_by_shape", [False, True])
def test_batch_mode_reads_lazy_readers_whole(tmp_path, group_by_shape):
    # batch mode predicts chips whole, so a reader from load_func is read whole, and its
    # profile is the default meta, as in tiled mode
    data = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    src = tmp_path / "in.tif"
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:32750",
        "transform": Affine(10, 0, 500000, 0, -10, 6000000),
    }
    with rio.open(src, "w", **profile) as dst:
        dst.write(data)
    from manypred import predict_batch

    out = predict_batch(
        lambda x: x,
        [src],
        load_func=RasterioReader,
        group_by_shape=group_by_shape,
        output="logits",
        progress=False,
        device="cpu",
    )
    np.testing.assert_array_equal(np.asarray(out[0] if group_by_shape else out[0]), data)
    paths = predict_files(
        lambda x: x,
        [src],
        load_func=RasterioReader,
        mode="batch",
        output_dir=tmp_path / "out",
        output="logits",
        progress=False,
        device="cpu",
    )
    with rio.open(paths[0]) as written:
        np.testing.assert_array_equal(written.read(), data)
        assert written.transform == profile["transform"] and written.crs == profile["crs"]
