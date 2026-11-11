"""load_raster and stack_rasters against real GeoTIFFs."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

rio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

from manypred import predict_files  # noqa: E402
from manypred.geo import RasterioReader, load_raster, stack_rasters  # noqa: E402
from manypred.lazy import reader_profile  # noqa: E402

LEFT, TOP = 400000, 6500000
RNG = np.random.default_rng(0)


def write_tif(
    path: Path,
    array: np.ndarray,
    res: float = 10,
    crs="EPSG:32750",
    left=LEFT,
    transform=None,
    nodata=None,
):
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": crs,
        "transform": transform or Affine(res, 0, left, 0, -res, TOP),
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path


def test_load_raster_native_and_band_order(tmp_path):
    data = RNG.integers(0, 10000, (4, 48, 64), dtype=np.uint16)
    path = write_tif(tmp_path / "a.tif", data)
    array, profile = load_raster(path)
    np.testing.assert_array_equal(array, data)
    assert (profile["height"], profile["width"], profile["count"]) == (48, 64, 4)
    assert profile["transform"] == Affine(10, 0, LEFT, 0, -10, TOP)
    picked, profile = load_raster(path, bands=[3, 1])
    np.testing.assert_array_equal(picked, data[[2, 0]])
    assert profile["count"] == 2 and profile["dtype"] == "uint16"
    assert not {"blockxsize", "driver", "compress"} & set(profile)  # the source's format


def test_load_raster_resampled_keeps_the_extent(tmp_path):
    data = RNG.random((2, 48, 64)).astype(np.float32)
    path = write_tif(tmp_path / "a.tif", data)
    coarse, profile = load_raster(path, resolution=20, resampling="average")
    assert coarse.shape == (2, 24, 32)
    expected = data.reshape(2, 24, 2, 32, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(coarse, expected, rtol=1e-5)
    assert profile["transform"] == Affine(20, 0, LEFT, 0, -20, TOP)
    assert (profile["height"], profile["width"]) == (24, 32)
    fine, profile = load_raster(path, resolution=5, resampling="nearest")
    np.testing.assert_array_equal(fine, data.repeat(2, axis=1).repeat(2, axis=2))
    assert profile["transform"] == Affine(5, 0, LEFT, 0, -5, TOP)


def test_load_raster_errors(tmp_path):
    path = write_tif(tmp_path / "a.tif", np.zeros((2, 8, 8), np.uint8))
    with pytest.raises(ValueError, match="bands 1-2"):
        load_raster(path, bands=[3])
    with pytest.raises(ValueError, match="Unknown resampling"):
        load_raster(path, resolution=20, resampling="bicubical")
    with pytest.raises(ValueError, match="positive"):
        load_raster(path, resolution=0)
    with pytest.raises(ValueError, match="no bands"):
        load_raster(path, bands=[])
    with pytest.raises(ValueError, match="Unknown resampling"):
        load_raster(path, resampling="bicubical")  # checked even without resizing
    rotated = write_tif(
        tmp_path / "r.tif",
        np.zeros((1, 8, 8), np.uint8),
        transform=Affine(10, 2, LEFT, 2, -10, TOP),
    )
    with pytest.raises(ValueError, match="rotation"):
        load_raster(rotated, resolution=20)


def test_load_raster_rounded_size_keeps_the_extent(tmp_path):
    """50 px of 10 m at 30 m: 17 px of about 29.4 m, the extent unchanged."""
    path = write_tif(tmp_path / "a.tif", np.ones((1, 50, 50), np.float32), nodata=-9999)
    data, profile = load_raster(path, resolution=30)
    assert data.shape == (1, 17, 17) and profile["nodata"] == -9999
    t = profile["transform"]
    assert t.c == LEFT and t.f == TOP
    assert t.a * 17 == pytest.approx(500) and t.e * 17 == pytest.approx(-500)


def test_stack_rasters_mixed_resolutions(tmp_path):
    """One file per band at 10, 20 and 60 m, as in a Sentinel-2 product."""
    b10 = RNG.integers(0, 10000, (1, 120, 120), dtype=np.uint16)
    b20 = RNG.integers(0, 10000, (1, 60, 60), dtype=np.uint16)
    b60 = RNG.integers(0, 10000, (2, 20, 20), dtype=np.uint16)
    paths = [
        write_tif(tmp_path / "b10.tif", b10, 10),
        write_tif(tmp_path / "b20.tif", b20, 20),
        write_tif(tmp_path / "b60.tif", b60, 60),
    ]
    data, profile = stack_rasters(paths, resampling="nearest")  # the finest: 10 m
    assert data.shape == (4, 120, 120) and data.dtype == np.uint16
    np.testing.assert_array_equal(data[0], b10[0])
    np.testing.assert_array_equal(data[1], b20[0].repeat(2, 0).repeat(2, 1))
    np.testing.assert_array_equal(data[2:], b60.repeat(6, 1).repeat(6, 2))
    assert profile["count"] == 4 and profile["transform"] == Affine(10, 0, LEFT, 0, -10, TOP)
    coarse, profile = stack_rasters(paths, resolution=20, resampling="nearest")
    assert coarse.shape == (4, 60, 60)
    np.testing.assert_array_equal(coarse[1], b20[0])
    assert profile["transform"] == Affine(20, 0, LEFT, 0, -20, TOP)


def test_stack_rasters_rejects_other_grids(tmp_path):
    a = write_tif(tmp_path / "a.tif", np.zeros((1, 16, 16), np.uint8))
    shifted = write_tif(tmp_path / "b.tif", np.zeros((1, 16, 16), np.uint8), left=LEFT + 50)
    other_crs = write_tif(tmp_path / "c.tif", np.zeros((1, 16, 16), np.uint8), crs="EPSG:32751")
    with pytest.raises(ValueError, match="same extent"):
        stack_rasters([a, shifted])
    with pytest.raises(ValueError, match="EPSG:32751"):
        stack_rasters([a, other_crs])
    with pytest.raises(ValueError, match="at least one"):
        stack_rasters([])
    nudged = write_tif(tmp_path / "d.tif", np.zeros((1, 16, 16), np.uint8), left=LEFT + 4)
    with pytest.raises(ValueError, match="same extent"):
        stack_rasters([a, nudged])  # under half a pixel off: would stack misregistered
    south_up = write_tif(
        tmp_path / "e.tif",
        np.zeros((1, 16, 16), np.uint8),
        transform=Affine(10, 0, LEFT, 0, 10, TOP - 160),
    )
    with pytest.raises(ValueError, match="north-up"):
        stack_rasters([a, south_up])  # same bounds, rows reversed


def test_stack_rasters_nodata(tmp_path):
    a = write_tif(tmp_path / "a.tif", np.ones((1, 8, 8), np.uint16), nodata=0)
    b = write_tif(tmp_path / "b.tif", np.ones((1, 8, 8), np.uint16), nodata=0)
    c = write_tif(tmp_path / "c.tif", np.ones((1, 8, 8), np.float32), nodata=-9999)
    assert stack_rasters([a, b])[1]["nodata"] == 0
    data, profile = stack_rasters([a, c])
    assert profile["nodata"] is None and data.dtype == np.float32 and profile["dtype"] == "float32"


def test_load_raster_as_load_func(tmp_path):
    """The prediction lands on the resampled grid."""
    data = RNG.integers(1, 10000, (3, 64, 96), dtype=np.uint16)
    path = write_tif(tmp_path / "scene.tif", data)
    [out] = predict_files(
        lambda x: x[:, :1],
        [path],
        load_func=lambda p: load_raster(p, resolution=20),
        output_dir=tmp_path / "out",
        patch_size=32,
        overlap=8,
        device="cpu",
        progress=False,
    )
    with rio.open(out) as src:
        assert (src.height, src.width) == (32, 48)
        assert src.transform == Affine(20, 0, LEFT, 0, -20, TOP)
        assert src.crs == "EPSG:32750"


def test_source_grid_writes_at_the_source_resolution(tmp_path):
    """grid="source": predict at 20 m, write the mask back on the file's 10 m grid."""
    data = RNG.integers(1, 10000, (3, 64, 96), dtype=np.uint16)
    data[:, :, :8] = 0  # nodata strip, for the mask band (nearest: no blending into it)
    path = write_tif(tmp_path / "scene.tif", data)
    array, profile = load_raster(path, resolution=20, grid="source")
    assert array.shape == (3, 32, 48)
    assert (profile["height"], profile["width"]) == (64, 96)
    assert profile["transform"] == Affine(10, 0, LEFT, 0, -10, TOP)
    kw = {"patch_size": 32, "overlap": 8, "device": "cpu", "progress": False, "nodata": 0}
    [fine_path] = predict_files(
        lambda x: x[:, :1],
        [path],
        load_func=lambda p: load_raster(p, resolution=20, resampling="nearest", grid="source"),
        output_dir=tmp_path / "source",
        **kw,
    )
    [coarse_path] = predict_files(
        lambda x: x[:, :1],
        [path],
        load_func=lambda p: load_raster(p, resolution=20, resampling="nearest"),
        output_dir=tmp_path / "array",
        **kw,
    )
    with rio.open(fine_path) as fine, rio.open(coarse_path) as coarse:
        assert (fine.height, fine.width) == (64, 96)
        assert fine.transform == Affine(10, 0, LEFT, 0, -10, TOP)
        assert (coarse.height, coarse.width) == (32, 48)
        expected = coarse.read(1).repeat(2, 0).repeat(2, 1)  # nearest neighbour
        np.testing.assert_array_equal(fine.read(1), expected)
        mask = fine.read_masks(1)
        assert (mask[:, :8] == 0).all() and (mask[:, 8:] == 255).all()
    stacked, profile = stack_rasters([path], resolution=20, grid="source")
    assert stacked.shape == (3, 32, 48) and (profile["height"], profile["width"]) == (64, 96)
    with pytest.raises(ValueError, match="'array' or 'source'"):
        load_raster(path, grid="other")


def test_geo_options_are_keyword_only_and_bands_checked(tmp_path):
    from manypred.geo import RasterioReader, write_geotiff

    path = write_tif(tmp_path / "a.tif", np.zeros((2, 8, 8), np.uint8))
    with pytest.raises(TypeError):
        load_raster(path, [1])
    with pytest.raises(TypeError):
        stack_rasters([path], 20)
    with pytest.raises(TypeError):
        write_geotiff(tmp_path / "o.tif", np.zeros((1, 8, 8), np.uint8), None, None)
    assert RasterioReader(path).shape == (2, 8, 8)
    assert RasterioReader(path, bands=[2]).shape == (1, 8, 8)
    with pytest.raises(ValueError, match="no bands"):
        RasterioReader(path, bands=[])
    with pytest.raises(ValueError, match="bands 1-2"):
        RasterioReader(path, bands=[0])


def test_write_geotiff_leaves_nothing_at_the_path_when_it_fails(tmp_path, monkeypatch):
    from manypred import geo

    array = np.zeros((1, 16, 16), np.float32)
    path = tmp_path / "x_pred.tif"
    real = geo.os.replace
    monkeypatch.setattr(geo.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        geo.write_geotiff(path, array, valid=np.ones((16, 16), bool))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(OSError, match="disk full"):
        geo.GeoTIFFWriter()(array, None, "x.tif", tmp_path)
    with pytest.raises(OSError, match="disk full"):
        geo.GeoTIFFWriter()(np.zeros(3), None, "x.tif", tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert geo.GeoTIFFWriter().existing("x.tif", tmp_path) is None
    monkeypatch.setattr(geo.os, "replace", real)
    geo.write_geotiff(path, array, valid=np.ones((16, 16), bool))
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # the mask stays inside the file
    with rio.open(path) as src:
        assert src.read_masks(1).min() == 255


def test_block_writer_keeps_its_rows_at_a_partial_path_until_closed(tmp_path):
    from manypred import geo

    path = tmp_path / "x_pred.tif"
    writer = geo.GeoTIFFWriter().open(None, "x.tif", tmp_path)
    writer.write("", 0, np.zeros((1, 8, 16), np.float32), (1, 16, 16), None)
    assert geo.GeoTIFFWriter().existing("x.tif", tmp_path) is None  # as if killed here
    writer.write("", 8, np.ones((1, 8, 16), np.float32), (1, 16, 16), None)
    assert writer.close() == path
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    with rio.open(path) as src:
        assert src.read(1)[15, 0] == 1
    writer = geo.GeoTIFFWriter().open(None, "y.tif", tmp_path)
    writer.write("", 0, np.zeros((1, 8, 16), np.float32), (1, 16, 16), None)
    writer.abort()
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_masks_stay_inside_the_file_whatever_the_environment(tmp_path):
    from manypred import geo

    valid = np.ones((16, 16), bool)
    with rio.Env(GDAL_TIFF_INTERNAL_MASK="NO"):
        geo.write_geotiff(tmp_path / "a.tif", np.zeros((1, 16, 16), np.float32), valid=valid)
        writer = geo.GeoTIFFWriter().open(None, "b.tif", tmp_path)
        writer.write("", 0, np.zeros((1, 16, 16), np.float32), (1, 16, 16), valid)
        writer.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tif", "b_pred.tif"]


class _Double(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return x * self.scale


KW = {"device": "cpu", "progress": False, "patch_size": 16, "overlap": 4, "output": "logits"}


def _scene(tmp_path):
    return write_tif(tmp_path / "scene.tif", RNG.random((2, 40, 48)).astype(np.float32))


def test_reader_alone_keeps_the_grid(tmp_path):
    path = _scene(tmp_path)
    out = predict_files(
        _Double(), [path], load_func=RasterioReader, output_dir=tmp_path / "o", **KW
    )
    with rio.open(path) as src, rio.open(out[0]) as dst:
        assert dst.crs == src.crs and dst.transform == src.transform
        assert (dst.height, dst.width) == (src.height, src.width)


def test_explicit_meta_wins_over_the_reader_profile(tmp_path):
    seen = []

    def save(prediction, meta, item, output_dir):
        seen.append(meta)
        return item

    path = _scene(tmp_path)
    predict_files(
        _Double(),
        [path],
        load_func=lambda p: (RasterioReader(p), {"mine": 1}),
        save_func=save,
        **KW,
    )
    predict_files(_Double(), [path], load_func=RasterioReader, save_func=save, **KW)
    predict_files(_Double(), [path], load_func=lambda p: load_raster(p)[0], save_func=save, **KW)
    assert seen[0] == {"mine": 1} and seen[1]["crs"] == "EPSG:32750" and seen[2] is None


class _NoSpecTwoInputs(torch.nn.Module):
    def __init__(self, spec=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.spec = spec

    def forward(self, x):
        return next(v for v in x.values() if v.ndim == 4) * self.scale

    def tiling_spec(self):
        return self.spec or {}


def _meta_seen(model, inputs):
    seen = []

    def save(prediction, meta, item, output_dir):
        seen.append(meta)

    predict_files(model, ["x"], load_func=lambda item: inputs, save_func=save, **KW)
    return seen[0]


def test_only_the_primary_inputs_profile_is_used(tmp_path):
    coarse = write_tif(tmp_path / "coarse.tif", RNG.random((1, 20, 24)).astype(np.float32), res=20)
    fine = np.ones((1, 40, 48), np.float32)
    inputs = {"fine": fine, "coarse": RasterioReader(coarse)}
    assert _meta_seen(_NoSpecTwoInputs(), inputs) is None
    inputs = {"meta": np.ones(3, np.float32), "coarse": RasterioReader(coarse)}
    assert _meta_seen(_NoSpecTwoInputs(), inputs)["height"] == 20


def test_the_specs_primary_input_gives_the_profile(tmp_path):
    aux = write_tif(tmp_path / "aux.tif", RNG.random((1, 40, 48)).astype(np.float32), left=LEFT)
    main = write_tif(
        tmp_path / "main.tif", RNG.random((1, 40, 48)).astype(np.float32), left=LEFT + 5000
    )
    spec = {"inputs": {"main": {"kind": "raster"}, "aux": {"kind": "raster"}}}
    inputs = {"aux": RasterioReader(aux), "main": RasterioReader(main)}  # dict order disagrees
    meta = _meta_seen(_NoSpecTwoInputs(spec), inputs)
    assert meta["transform"].c == LEFT + 5000


def test_reader_profile_has_no_source_layout(tmp_path):
    path = _scene(tmp_path)  # tiled and deflated
    profile = RasterioReader(path, bands=[2]).profile
    for key in ("driver", "tiled", "blockxsize", "blockysize", "compress", "interleave"):
        assert key not in profile
    assert profile["count"] == 1 and profile["crs"] == "EPSG:32750" and profile["height"] == 40


def test_lazy_output_of_a_tiled_source_matches_the_eager_layout(tmp_path):
    path = _scene(tmp_path)
    lazy = predict_files(
        _Double(), [path], load_func=RasterioReader, output_dir=tmp_path / "l", **KW
    )
    eager = predict_files(_Double(), [path], load_func=load_raster, output_dir=tmp_path / "e", **KW)
    with rio.open(lazy[0]) as a, rio.open(eager[0]) as b:
        assert not a.is_tiled and a.block_shapes == b.block_shapes
        np.testing.assert_array_equal(a.read(), b.read())


def test_lazy_output_ignores_a_tiled_layout_in_any_profile(tmp_path):
    path = _scene(tmp_path)

    def load(p):
        with rio.open(p) as src:
            return RasterioReader(p), src.profile

    out = predict_files(_Double(), [path], load_func=load, output_dir=tmp_path / "o", **KW)
    with rio.open(out[0]) as dst:
        assert not dst.is_tiled and dst.crs == "EPSG:32750"


def test_reader_profile_of_another_grid_is_not_used():
    class Resampling:
        shape = (1, 40, 48)
        profile = {"height": 80, "width": 96, "crs": "EPSG:32750"}

        def read(self, window):
            raise AssertionError("not read here")

    assert reader_profile(Resampling()) is None


def test_uppercase_layout_options_are_ignored_for_row_blocks(tmp_path):
    path = _scene(tmp_path)
    layout = {"TILED": "YES", "BLOCKXSIZE": 16, "BLOCKYSIZE": 16}

    def load(p):
        reader = RasterioReader(p)
        return reader, {**reader.profile, **layout}

    out = predict_files(_Double(), [path], load_func=load, output_dir=tmp_path / "o", **KW)
    with rio.open(out[0]) as dst:
        assert not dst.is_tiled


def _nodata_scene(tmp_path, name="declared.tif", nodata=255.0):
    array = RNG.random((1, 32, 32)).astype(np.float32) * 100
    return write_tif(tmp_path / name, array, nodata=nodata)


def _nodata_warnings(record):
    return [w for w in record if "declares nodata" in str(w.message)]


def _save_nothing(prediction, meta, item, output_dir):
    return item


FILE_KW = {"device": "cpu", "progress": False, "patch_size": 32, "overlap": 0, "output": "logits"}


@pytest.mark.parametrize("mode", ["tiled", "batch"])
@pytest.mark.parametrize("load", [load_raster, RasterioReader])
def test_declared_nodata_warns_once_at_the_caller(tmp_path, mode, load):
    if mode == "batch" and load is RasterioReader:
        pytest.skip("batch mode takes arrays")
    paths = [_nodata_scene(tmp_path, f"{i}.tif") for i in range(2)]
    with pytest.warns(UserWarning, match="declares nodata=255") as record:
        predict_files(
            _Double(), paths, load_func=load, save_func=_save_nothing, mode=mode, **FILE_KW
        )
    found = _nodata_warnings(record)
    assert len(found) == 1 and str(paths[0]) in str(found[0].message)
    assert found[0].filename == __file__


@pytest.mark.parametrize(
    "nodata, declared",
    [(255.0, 255.0), (0, 255.0), (float("nan"), 255.0), (None, float("nan")), (None, None)],
)
def test_declared_nodata_is_quiet_when_settled(tmp_path, nodata, declared):
    path = _nodata_scene(tmp_path, nodata=declared)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_files(
            _Double(),
            [path],
            load_func=load_raster,
            save_func=_save_nothing,
            nodata=nodata,
            **FILE_KW,
        )
    assert not _nodata_warnings(record)


def test_declared_nodata_only_warns_it_changes_nothing(tmp_path):
    path = _nodata_scene(tmp_path)
    with pytest.warns(UserWarning, match="declares nodata"):
        out = predict_files(
            _Double(), [path], load_func=load_raster, output_dir=tmp_path / "o", **FILE_KW
        )
    with rio.open(out[0]) as dst, rio.open(path) as src:
        np.testing.assert_allclose(dst.read(), src.read() * 2)
        assert dst.read_masks(1).all()


@pytest.mark.parametrize(
    "value", [np.array([0, 1]), "quality-code", float("inf"), -float("inf"), True, [255]]
)
def test_meta_nodata_that_is_not_a_finite_number_is_ignored(value):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_files(
            _Double(),
            ["a"],
            load_func=lambda item: (np.ones((1, 32, 32), np.float32), {"nodata": value}),
            save_func=_save_nothing,
            **FILE_KW,
        )
    assert not _nodata_warnings(record)


@pytest.mark.parametrize("value", [255, 255.0, np.float32(255), np.uint8(255)])
def test_meta_nodata_of_any_number_type_warns(value):
    with pytest.warns(UserWarning, match="declares nodata=255"):
        predict_files(
            _Double(),
            ["a"],
            load_func=lambda item: (np.ones((1, 32, 32), np.float32), {"nodata": value}),
            save_func=_save_nothing,
            **FILE_KW,
        )
