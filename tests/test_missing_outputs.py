"""Missing pixels in outputs: NaN in float outputs, wherever patches fell; class 0 in class
outputs, unless the model returned NaN where there is data, which is an error. Float16 mosaics
stay accurate where the blend weights are tiny."""

import numpy as np
import pytest
import torch

from manypred import predict_array, predict_batch, predict_files
from manypred.blend import BandMosaic, Mosaic, gradient_weights
from manypred.lazy import ArrayReader

CPU = {"device": "cpu", "progress": False}
TILED = {"patch_size": 32, "overlap": 8, **CPU}


def conv(classes=2):
    torch.manual_seed(0)
    return torch.nn.Conv2d(3, classes, 1)


def scene():
    """Nodata (all bands 0) in the top left corner, which some patches cover and some skip."""
    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    image[:, :50, :50] = 0
    return image


class FailsLater(torch.nn.Module):
    """Returns NaN from its second call on, or at chosen pixels of every call."""

    def __init__(self, classes=2, only_where_zero=False, one_class=False):
        super().__init__()
        self.conv = conv(classes)
        self.calls = 0
        self.only_where_zero, self.one_class = only_where_zero, one_class

    def forward(self, x):
        self.calls += 1
        out = self.conv(x)
        if self.only_where_zero:
            return torch.where((x == 0).all(1, keepdim=True), torch.nan, out)
        if self.calls > 1:
            if self.one_class:
                out = out.clone()
                out[:, 0] = torch.nan
                return out
            return out * torch.nan
        return out


@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("merge", ["blend", "cut"])
def test_missing_pixels_are_nan_wherever_patches_fall(shift, merge):
    out = predict_array(conv(), scene(), nodata=0, shift_patches=shift, merge=merge, **TILED)
    assert np.isnan(out[:, :50, :50]).all()
    assert np.isfinite(out[:, 50:, 50:]).all()


def test_reduced_resolution_output():
    class Quarter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv()

        def forward(self, x):
            return self.conv(x)[..., ::4, ::4]

    out = predict_array(Quarter(), scene(), nodata=0, **TILED)
    assert np.isnan(out[:, :12, :12]).all()
    assert np.isfinite(out[:, 13:, 13:]).all()


def test_callable_output_is_masked_too():
    out = predict_array(conv(), scene(), nodata=0, output=lambda x: x * 2, **TILED)
    assert np.isnan(out[:, :50, :50]).all()


def test_batch_partial_and_empty_chips_agree():
    image = scene()
    chips = np.stack([image[:, 40:72, 40:72], image[:, :32, :32]])
    out = predict_batch(conv(), chips, nodata=0, **CPU)
    assert np.isnan(out[0, :, :10, :10]).all()
    assert np.isfinite(out[0, :, 10:, 10:]).all()
    assert np.isnan(out[1]).all()


def test_mask_off_keeps_what_the_model_returned():
    out = predict_array(conv(), scene(), nodata=0, mask_nodata=False, shift_patches=False, **TILED)
    assert np.isfinite(out[:, 40:50, 40:50]).all()  # covered by a patch that also has data


def test_classes_at_missing_pixels_are_zero():
    out = predict_array(conv(), scene(), nodata=0, output="argmax", **TILED)
    assert (out[:, :50, :50] == 0).all()


def test_lazy_scene_written_in_blocks(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    written = predict_files(
        conv(),
        ["a.tif"],
        load_func=lambda item: ArrayReader(scene()),
        output_dir=tmp_path,
        nodata=0,
        **TILED,
    )
    with rasterio.open(written[0]) as file:
        out = file.read()
    eager = predict_array(conv(), scene(), nodata=0, **TILED)
    np.testing.assert_array_equal(out, eager)
    assert np.isnan(out[:, :50, :50]).all()


@pytest.mark.parametrize("merge", ["blend", "cut"])
@pytest.mark.parametrize("one_class", [False, True])
def test_argmax_of_model_nan_is_an_error(merge, one_class):
    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    with pytest.raises(RuntimeError, match="returned NaN at .* pixels with data"):
        predict_array(FailsLater(one_class=one_class), image, output="argmax", merge=merge, **TILED)


def test_argmax_of_model_nan_is_an_error_at_reduced_resolution():
    class Quarter(FailsLater):
        def forward(self, x):
            return super().forward(x)[..., ::4, ::4]

    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    with pytest.raises(RuntimeError, match="returned NaN"):
        predict_array(Quarter(), image, output="argmax", **TILED)


def test_argmax_names_the_output():
    class Heads(FailsLater):
        def forward(self, x):
            return {"clouds": super().forward(x)}

    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    with pytest.raises(RuntimeError, match="'clouds'"):
        predict_array(Heads(), image, output="argmax", **TILED)


@pytest.mark.parametrize("group", [False, True])
def test_batch_argmax_of_model_nan_is_an_error(group):
    chips = np.random.default_rng(0).random((4, 3, 16, 16), dtype=np.float32) + 0.1
    with pytest.raises(RuntimeError, match="returned NaN"):
        predict_batch(
            FailsLater(), list(chips), batch_size=1, output="argmax", group_by_shape=group, **CPU
        )


def test_batch_image_level_argmax_of_model_nan_is_an_error():
    model = lambda x: torch.full((len(x), 3), torch.nan)  # noqa: E731
    chips = np.ones((2, 3, 16, 16), dtype=np.float32)
    with pytest.raises(RuntimeError, match="returned NaN"):
        predict_batch(model, chips, output="argmax", **CPU)


@pytest.mark.parametrize("mask", [True, False])
def test_nan_only_over_nodata_is_not_an_error(mask):
    model = FailsLater(only_where_zero=True)
    out = predict_array(model, scene(), nodata=0, output="argmax", mask_nodata=mask, **TILED)
    assert (out[:, :50, :50] == 0).all()
    chips = np.stack([scene()[:, :32, :32], scene()[:, 40:72, 40:72]])
    predict_batch(model, chips, nodata=0, output="argmax", mask_nodata=mask, **CPU)


def test_all_nodata_chip_with_argmax_is_not_an_error():
    chips = np.zeros((2, 3, 16, 16), dtype=np.float32)
    out = predict_batch(conv(), chips, nodata=0, output="argmax", **CPU)
    assert (out == 0).all()


def test_float_outputs_keep_what_the_model_returned():
    """NaN shows in a float output, and a model may mean it, so it isn't an error."""
    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    out = predict_array(FailsLater(), image, **TILED)
    assert np.isnan(out).any()


def write(path, image):
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", height=128, width=128, count=3, dtype="float32"
    ) as f:
        f.write(image)


@pytest.mark.parametrize("lazy", [False, True])
def test_failed_scene_is_not_marked_done(tmp_path, lazy):
    rasterio = pytest.importorskip("rasterio")
    from manypred.geo import RasterioReader

    """A rerun with overwrite=False keeps the finished scene and retries the failed one."""
    image = np.random.default_rng(0).random((3, 128, 128), dtype=np.float32) + 0.1
    good, bad = tmp_path / "good.tif", tmp_path / "bad.tif"
    write(good, image)
    write(bad, image)
    out = tmp_path / "out"

    def load(path):
        return RasterioReader(path) if lazy else rasterio.open(path).read()

    class FailsOnSecondScene(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv()
            self.scene = None
            self.limit = 0

        def forward(self, x):
            self.limit += 1
            return self.conv(x) * (torch.nan if self.limit > 25 else 1.0)

    kwargs = {"load_func": load, "output_dir": out, "output": "argmax", **TILED}
    with pytest.raises(RuntimeError, match="returned NaN"):
        predict_files(FailsOnSecondScene(), [good, bad], **kwargs)
    assert (out / "good_pred.tif").exists()
    assert not (out / "bad_pred.tif").exists()
    loaded = []

    def counted(path):
        loaded.append(path.name)
        return load(path)

    predict_files(conv(), [good, bad], **{**kwargs, "load_func": counted}, overwrite=False)
    assert loaded == ["bad.tif"]
    assert (out / "bad_pred.tif").exists()


@pytest.mark.parametrize("cls", [Mosaic, BandMosaic])
@pytest.mark.parametrize("patch, overlap", [(1000, 300), (2048, 1024), (4096, 2048)])
def test_float16_mosaic_is_accurate_at_tiny_weights(cls, patch, overlap):
    mosaic = cls(1, patch, patch, gradient_weights(patch, overlap), "cpu", torch.float16)
    mosaic.add(torch.full((1, 1, patch, patch), 0.37), [(0, patch, 0, patch)])
    if cls is BandMosaic:
        _, sums, weight = mosaic.take(patch)
        values = mosaic.merged(sums, weight)
    else:
        values = mosaic.blended()
    assert float((values - 0.37).abs().max()) < 0.37 * 0.005


def test_float16_weights_never_round_to_zero():
    weights = torch.full((4, 4), 1e-8)
    mosaic = Mosaic(1, 4, 4, weights, "cpu", torch.float16)
    assert (mosaic.weights > 0).all()
    mosaic.add(torch.ones((1, 1, 4, 4)), [(0, 4, 0, 4)])
    assert torch.isfinite(mosaic.blended()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_other_dtypes_keep_the_weights_exactly(dtype):
    weights = gradient_weights(512, 256)
    mosaic = Mosaic(1, 512, 512, weights, "cpu", dtype)
    assert torch.equal(mosaic.weights, weights.to(dtype))
