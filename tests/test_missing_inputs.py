"""Missing input values (NaN, infinity, or ``nodata``) reach the model as 0, with or without
normalisation."""

import warnings

import numpy as np
import pytest
import torch

from manypred import normalize_numpy, predict_array, predict_batch, predict_files
from manypred.lazy import ArrayReader

CPU = {"device": "cpu", "progress": False}
TILED = {"patch_size": 32, "overlap": 8, **CPU}
NO_STEPS = [None, False, {"type": "none"}]


class Spy(torch.nn.Module):
    """A 3 x 3 convolution, which spreads NaN to its neighbours, keeping what it was given."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.conv(x)


def scene(fill=np.nan):
    image = np.random.default_rng(0).random((3, 64, 80), dtype=np.float32) + 0.1
    image[:, 10:14, 10:14] = fill
    return image


def assert_only_missing_is_nan(out):
    """NaN at the missing pixels of :func:`scene`, as any nodata pixel, and it didn't spread."""
    expected = np.zeros(out.shape[-2:], dtype=bool)
    expected[10:14, 10:14] = True
    np.testing.assert_array_equal(np.isnan(out).any(axis=0), expected)


@pytest.mark.parametrize("normalize", NO_STEPS)
def test_tiled_nan_reaches_the_model_as_zero(normalize):
    model = Spy()
    out = predict_array(model, scene(), normalize=normalize, **TILED)
    assert all(torch.isfinite(x).all() for x in model.seen)
    assert_only_missing_is_nan(out)


@pytest.mark.parametrize("normalize", NO_STEPS)
def test_batch_nan_reaches_the_model_as_zero(normalize):
    model = Spy()
    out = predict_batch(model, scene()[None, :, :32, :32], normalize=normalize, **CPU)
    assert torch.isfinite(model.seen[0]).all()
    assert_only_missing_is_nan(out[0])


def test_no_steps_all_give_the_same_result():
    results = [predict_array(Spy(), scene(), normalize=n, **TILED) for n in NO_STEPS]
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


@pytest.mark.parametrize("normalize", NO_STEPS)
def test_nodata_value_reaches_the_model_as_zero(normalize):
    model = Spy()
    out = predict_array(model, scene(fill=-9999), nodata=-9999, normalize=normalize, **TILED)
    assert min(float(x.min()) for x in model.seen) == 0
    assert_only_missing_is_nan(out)


def test_nan_in_one_band_is_zero_there_and_the_pixel_stays_valid():
    image = scene(fill=0.5)
    image[0, 10:14, 10:14] = np.nan
    model = Spy()
    out = predict_array(model, image, patch_size=64, overlap=0, **CPU)
    seen = model.seen[0][0]
    assert (seen[0, 10:14, 10:14] == 0).all()
    assert (seen[1:, 10:14, 10:14] == 0.5).all()
    assert (out[:, 11, 11] != 0).all()


def test_values_without_anything_missing_are_unchanged():
    model = Spy()
    image = np.random.default_rng(0).integers(0, 65535, (3, 32, 32), dtype=np.uint16)
    predict_array(model, image, patch_size=32, overlap=0, **CPU)
    np.testing.assert_array_equal(model.seen[0][0].numpy(), image.astype(np.float32))


def test_lazy_scene():
    model = Spy()
    out = predict_array(model, ArrayReader(scene()), **TILED)
    assert all(torch.isfinite(x).all() for x in model.seen)
    assert_only_missing_is_nan(out)


def test_files(tmp_path):
    model, saved = Spy(), []

    def save(prediction, meta, item, output_dir):
        saved.append(prediction)
        return item

    predict_files(model, ["a"], load_func=lambda item: scene(), save_func=save, **TILED)
    assert all(torch.isfinite(x).all() for x in model.seen)
    assert_only_missing_is_nan(saved[0])


class Inputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append({name: t.detach().clone() for name, t in x.items()})
        return x["a"][:, :1]


@pytest.mark.parametrize("normalize", [None, {"a": {"type": "scale", "value": 2}}])
def test_every_raster_input_with_or_without_its_own_step(normalize):
    half = scene()[:, ::2, ::2]
    series = np.stack([scene()] * 2)
    vector = np.array([1.0, np.nan], dtype=np.float32)
    model = Inputs()
    inputs = {"a": scene(), "half": half, "series": series, "vector": vector}
    predict_array(model, inputs, normalize=normalize, **TILED)
    for batch in model.seen:
        for name in ("a", "half", "series"):
            assert torch.isfinite(batch[name]).all(), name
        assert torch.isnan(batch["vector"]).any()  # vectors are passed as they are


def test_padding_past_the_scene_edge_is_zero():
    # a quarter-resolution head on a scene that isn't a multiple of 4 pads the last patches
    class Quarter(Spy):
        def forward(self, x):
            return super().forward(x)[..., ::4, ::4]

    model = Quarter()
    predict_array(model, scene()[:, :62, :78], nodata=float("nan"), **TILED)
    assert all(torch.isfinite(x).all() for x in model.seen)


def test_stored_normalisation_turned_off():
    class Stored(Spy):
        def tiling_spec(self):
            return {"normalize": {"type": "scale", "value": 2}}

    model = Stored()
    image = scene()
    predict_array(model, image, normalize=False, patch_size=64, overlap=0, **CPU)
    seen = model.seen[0][0].numpy()
    assert (seen[:, 10:14, 10:14] == 0).all()
    assert seen[0, 0, 0] == image[0, 0, 0]  # not scaled


def test_nan_inputs_dont_warn():
    model = Spy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        predict_batch(model, scene()[None, :, :32, :32], **CPU)
    assert model.conv.weight.dtype == torch.float32


INFINITE = [float("inf"), float("-inf")]
STEPS = [
    None,
    {"type": "none"},
    {"type": "scale", "value": 2},
    {"type": "zscore", "scope": "tile"},
    {"type": "zscore", "scope": "scene"},
    {"type": "percentile", "scope": "tile"},
    {"type": "percentile", "scope": "scene"},
]


@pytest.mark.parametrize("value", INFINITE)
@pytest.mark.parametrize("normalize", STEPS)
def test_infinity_is_missing_like_nan(value, normalize):
    """Same inputs, statistics and outputs as NaN at those pixels: an infinite value doesn't
    blank its patch or fail."""
    runs = {}
    for fill in (np.nan, value):
        model = Spy()
        out = predict_array(model, scene(fill=fill), normalize=normalize, **TILED)
        runs[fill] = out, model.seen
    nan_out, nan_seen = runs[np.nan]
    out, seen = runs[value]
    np.testing.assert_array_equal(out, nan_out)
    assert_only_missing_is_nan(out)
    for a, b in zip(seen, nan_seen, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("value", INFINITE)
def test_infinity_in_one_band_leaves_the_others(value):
    image = scene(fill=0.5)
    image[0, 10:14, 10:14] = value
    model = Spy()
    out = predict_array(
        model, image, patch_size=64, overlap=0, normalize={"type": "zscore", "scope": "tile"}, **CPU
    )
    seen = model.seen[0][0]
    assert (seen[0, 10:14, 10:14] == 0).all()
    assert (seen != 0).float().mean() > 0.9  # the patch isn't blanked
    assert np.isfinite(out).all()  # the pixel still has data in the other bands


def test_infinity_is_skipped_and_masked_as_nodata():
    image = np.full((3, 64, 80), 0.5, dtype=np.float32)
    image[:, :32, :] = float("inf")
    model = Spy()
    out = predict_array(model, image, patch_size=32, overlap=0, shift_patches=False, **CPU)
    assert len(model.seen) == 3  # the top row of patches is skipped
    assert np.isnan(out[:, :32]).all()
    assert np.isfinite(out[:, 32:]).all()


def test_infinity_in_a_batch_chip_and_normalize_functions():
    chip = scene(fill=float("inf"))[:, :32, :32]
    out = predict_batch(Spy(), chip[None], normalize={"type": "zscore", "scope": "tile"}, **CPU)
    assert_only_missing_is_nan(out[0])
    x = normalize_numpy(chip, {"type": "zscore", "scope": "scene"})
    assert np.isfinite(x).all() and (x[:, 10:14, 10:14] == 0).all()


def test_lazy_scene_with_infinity():
    image = scene(fill=float("inf"))
    kwargs = {"normalize": {"type": "zscore", "scope": "scene"}, **TILED}
    np.testing.assert_array_equal(
        predict_array(Spy(), ArrayReader(image), **kwargs), predict_array(Spy(), image, **kwargs)
    )


def test_adapt_patch_size_counts_infinity_as_missing():
    image = np.full((3, 64, 64), 0.5, dtype=np.float32)
    image[:, :40] = float("inf")
    with pytest.warns(UserWarning, match="Over 30%"):
        predict_array(Spy(), image, patch_size=48, overlap=8, adapt_patch_size=True, **CPU)
