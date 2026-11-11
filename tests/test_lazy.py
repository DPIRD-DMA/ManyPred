"""Lazy readers and block writing."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import SeededConv

import manypred.blend
from manypred import predict_array, predict_files
from manypred.lazy import ArrayReader, PackedMask, decimated, nodata_mask
from manypred.outputs import make_specs
from manypred.tiled import Tiler
from manypred.windows import nodata_pixels

RNG = np.random.default_rng(0)
KW = {"device": "cpu", "progress": False, "nodata": 0}


def scene(height=150, width=170):
    x = RNG.random((3, height, width)).astype(np.float32) + 0.1
    x[:, :20, :50] = 0
    x[:, 100:, 150:] = 0
    return x


class Heads(torch.nn.Module):
    def forward(self, x):
        dense = SeededConv()(x)
        return {"seg": dense, "quarter": F.avg_pool2d(dense, 4), "score": x.mean(dim=(2, 3))}


HEAD_KW = {"activation": {"score": "sigmoid"}, "output": {"seg": "argmax", "score": "probs"}}


def test_packed_mask_slices_like_the_array():
    mask = RNG.random((37, 53)) > 0.5
    packed = PackedMask(np.packbits(mask, axis=1), 53)
    for key in [
        (slice(3, 20), slice(5, 40)),
        (7, slice(0, 53)),
        (slice(0, 37), 11),
        (slice(30, 37), slice(50, 60)),
    ]:
        np.testing.assert_array_equal(packed[key], mask[key])
    assert packed.any() and packed.shape == (37, 53)
    np.testing.assert_array_equal(packed.unpacked(), mask)


def test_nodata_mask_from_reader_in_bands():
    x = scene()
    reader = ArrayReader(x)
    np.testing.assert_array_equal(nodata_mask(reader, 0).unpacked(), nodata_pixels(x, 0))


@pytest.mark.parametrize("shape", [(1, 1, 16_000_000), (1, 16_000_000, 1), (1, 3000, 5000)])
def test_decimated_stays_within_the_limit_on_any_shape(shape):
    sample = decimated(ArrayReader(np.zeros(shape, np.uint8)))
    assert sample.shape[-2] * sample.shape[-1] <= 4_000_000


def test_decimated_samples_every_kth_pixel():
    x = scene(400, 300)
    sample = decimated(ArrayReader(x), max_pixels=10_000)
    step = int(np.ceil(np.sqrt(400 * 300 / 10_000)))
    np.testing.assert_array_equal(sample, x[:, ::step, ::step])


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (SeededConv(), {"output": "argmax"}),
        (SeededConv(), {"activation": "softmax", "output": "probs"}),
        (Heads(), HEAD_KW),
    ],
)
def test_reader_gives_the_same_as_the_array(model, kwargs):
    x = scene()
    reference = predict_array(model, x, patch_size=48, overlap=16, **kwargs, **KW)
    lazy = predict_array(model, ArrayReader(x), patch_size=48, overlap=16, **kwargs, **KW)
    if isinstance(reference, dict):
        for name in ("seg", "quarter"):
            np.testing.assert_array_equal(lazy[name], reference[name])
        np.testing.assert_array_equal(lazy["score"].values, reference["score"].values)
    else:
        np.testing.assert_array_equal(lazy, reference)


class Collect:
    """A block writer assembling the blocks, recording each block's rows."""

    def __init__(self):
        self.arrays, self.written, self.blocks = {}, {}, []

    def write(self, name, top, block, shape, valid, factor=1):
        array = self.arrays.setdefault(name, np.zeros(shape, dtype=block.dtype))
        written = self.written.setdefault(name, np.zeros(shape[1], dtype=bool))
        rows = slice(top, top + block.shape[1])
        assert not written[rows].any(), "rows written twice"
        written[rows] = True
        array[:, rows] = block
        self.blocks.append((name, top, block.shape[1]))


def tiler(model, patch_size=48, overlap=16, **kwargs):
    from manypred.runner import Runner

    spec = {"activation": kwargs.pop("activation", None), "output": kwargs.pop("output", None)}
    runner = Runner(
        models=[model],
        device=torch.device("cpu"),
        dtype=torch.float32,
        normalizers={},
        nodata=0,
        primary="",
        make_specs=lambda names: make_specs(names, spec["activation"], spec["output"]),
    )
    return Tiler(
        runner,
        patch_size,
        overlap,
        2,
        torch.device("cpu"),
        True,
        False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (SeededConv(), {"output": "argmax"}),
        (SeededConv(), {}),
        (Heads(), HEAD_KW),
    ],
)
def test_blocks_assemble_to_the_in_memory_result(model, kwargs):
    x = scene(300, 170)
    expected, _ = tiler(model, **dict(kwargs)).predict(ArrayReader(x))
    writer = Collect()
    extras = tiler(model, **dict(kwargs)).predict_blocks(ArrayReader(x), writer)
    for name, array in writer.arrays.items():
        assert writer.written[name].all()
        np.testing.assert_array_equal(array, expected[name])
    assert set(writer.arrays) | set(extras) == set(expected)
    # written in several blocks, as rows were finished
    assert len([b for b in writer.blocks if b[0] == next(iter(writer.arrays))]) > 3


@pytest.mark.parametrize("height", [256, 1024, 4096])
def test_band_held_does_not_grow_with_the_scene(height, monkeypatch):
    held = []
    add = manypred.blend.BandMosaic.add

    def recording_add(self, predictions, windows):
        add(self, predictions, windows)
        held.append(self.bottom - self.top)

    monkeypatch.setattr(manypred.blend.BandMosaic, "add", recording_add)
    x = RNG.random((1, height, 64)).astype(np.float32) + 0.1
    tiler(torch.nn.Identity(), patch_size=32, overlap=8).predict_blocks(ArrayReader(x), Collect())
    assert max(held) <= 32 + 24  # one row of patches plus the next stride


def test_predict_files_with_a_reader_writes_the_same_files(tmp_path):
    rio = pytest.importorskip("rasterio")
    from affine import Affine

    from manypred.geo import RasterioReader

    x = (scene() * 1000).astype(np.uint16)
    path = tmp_path / "scene.tif"
    profile = {
        "driver": "GTiff",
        "height": x.shape[1],
        "width": x.shape[2],
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32750",
        "transform": Affine(10, 0, 400000, 0, -10, 6500000),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(x)

    def load_array(p):
        with rio.open(p) as src:
            return src.read(), src.profile

    def load_lazy(p):
        reader = RasterioReader(p)
        return reader, reader.profile

    kwargs = {"patch_size": 48, "overlap": 16, **HEAD_KW, **KW}
    [eager] = predict_files(
        Heads(), [path], load_func=load_array, output_dir=tmp_path / "a", **kwargs
    )
    [lazy] = predict_files(
        Heads(), [path], load_func=load_lazy, output_dir=tmp_path / "b", **kwargs
    )
    assert eager.name == lazy.name == "scene_pred.json"
    for name in ("seg", "quarter"):
        with (
            rio.open(tmp_path / "a" / f"scene_pred_{name}.tif") as a,
            rio.open(tmp_path / "b" / f"scene_pred_{name}.tif") as b,
        ):
            np.testing.assert_array_equal(a.read(), b.read())
            np.testing.assert_array_equal(a.read_masks(1), b.read_masks(1))
            # the eager file keeps its source's block size; row blocks use GDAL's default
            assert {**a.profile, "blockysize": 0} == {**b.profile, "blockysize": 0}
            if name == "quarter":
                assert a.shape == (38, 43) and a.transform.a == 40
    assert eager.read_text() == lazy.read_text()


def lazy_scene_file(tmp_path, rio):
    from affine import Affine

    x = (scene(96, 120) * 1000).astype(np.uint16)
    path = tmp_path / "scene.tif"
    profile = {
        "driver": "GTiff",
        "height": 96,
        "width": 120,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32750",
        "transform": Affine(10, 0, 400000, 0, -10, 6500000),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(x)
    return path


def test_block_writer_cleans_up_after_an_error(tmp_path):
    rio = pytest.importorskip("rasterio")
    from manypred.geo import RasterioReader

    path = lazy_scene_file(tmp_path, rio)
    calls = []

    def model(x):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("model failed")
        return SeededConv()(x)

    def load(p):
        reader = RasterioReader(p)
        return reader, reader.profile

    with pytest.raises(RuntimeError, match="model failed"):
        predict_files(
            model,
            [path],
            load_func=load,
            output_dir=tmp_path / "out",
            patch_size=32,
            overlap=8,
            **KW,
        )
    assert not (tmp_path / "out" / "scene_pred.tif").exists()  # no partial file left
    assert not rio.env.hasenv()  # the GDAL cache setting was undone


def test_tile_predictions_on_lazy_scenes_stay_packed():
    from manypred.lazy import PackedMask

    x = scene()
    kwargs = {"patch_size": 48, "overlap": 16, "output": "logits", **KW}
    lazy = predict_array(lambda t: t.mean(dim=(2, 3)), ArrayReader(x), **kwargs)
    eager = predict_array(lambda t: t.mean(dim=(2, 3)), x, **kwargs)
    assert isinstance(lazy.valid_map, PackedMask)
    np.testing.assert_allclose(lazy.reduce("mean"), eager.reduce("mean"), rtol=1e-6)


def test_banded_reduce_mean_matches_the_full_map():
    x = scene(300, 200)
    tp = predict_array(
        lambda t: t.mean(dim=(2, 3)), x, patch_size=48, overlap=16, output="logits", **KW
    )
    full = tp.to_map()
    keep = ~np.isnan(full[0]) & tp.valid_map
    np.testing.assert_allclose(
        tp.reduce("mean"), full[:, keep].mean(axis=1, dtype=np.float64), rtol=1e-6
    )
    np.testing.assert_array_equal(tp.to_map(rows=(40, 90)), full[:, 40:90])


def test_shape_changing_callable_output_fails_in_block_mode():
    x = scene()
    with pytest.raises(ValueError, match="pixel-wise"):
        tiler(SeededConv(), output=lambda t: t[..., ::2, :]).predict_blocks(
            ArrayReader(x), Collect()
        )


def test_export_profile_scales_only_on_the_predicted_grid():
    Affine = pytest.importorskip("affine").Affine

    from manypred.geo import export_profile

    profile = {"height": 100, "width": 100, "transform": Affine(10, 0, 0, 0, -10, 0)}
    reduced = np.zeros((1, 25, 25), np.uint8)
    out = export_profile(profile, reduced, grid=(100, 100))
    assert (out["height"], out["width"], out["transform"].a) == (25, 25, 40)
    # a loader resampled to 50 x 50 but returned the source profile: no guessing
    out = export_profile(profile, reduced, grid=(50, 50))
    assert (out["height"], out["width"], out["transform"].a) == (100, 100, 10)
