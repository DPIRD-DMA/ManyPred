"""Normalisation specs: each type against a numpy reference, nodata and NaN
ignored by statistics and filled with 0, tile vs scene scope, errors."""

import numpy as np
import pytest
import torch

from manypred import normalize_numpy, normalize_torch, predict_array, predict_batch
from manypred.normalize import validate_spec

RNG = np.random.default_rng(0)


def scene(nodata=0.0):
    """(3, 20, 30) positive float32 with nodata pixels (all bands), one nodata value in band 0
    only, and a NaN in band 1."""
    x = (RNG.random((3, 20, 30)) * 1000 + 1).astype(np.float32)
    x[:, :4, :5] = nodata
    x[0, 10, 10] = nodata
    x[1, 12, 12] = np.nan
    return x


def valid_values(x, band, nodata=0.0):
    values = x[band]
    keep = ~np.isnan(values) & (values != nodata)
    return values[keep], keep


def zscore_ref(x, nodata=0.0):
    out = np.zeros_like(x)
    for band in range(len(x)):
        values, keep = valid_values(x, band, nodata)
        out[band][keep] = (values - values.mean()) / values.std()
    return out


def check(spec, x, expected, nodata=0.0, rtol=1e-5, atol=1e-5):
    out = normalize_numpy(x, spec, nodata=nodata)
    assert out.dtype == np.float32 and out.shape == x.shape
    np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)
    # torch, batched, gives the same
    batch = normalize_torch(torch.from_numpy(np.stack([x, x])), spec, nodata=nodata)
    np.testing.assert_allclose(batch[1].numpy(), out, rtol=1e-6, atol=1e-6)
    return out


def invalid(x, nodata=0.0):
    return np.isnan(x) | (x == nodata)


def test_none():
    x = scene()
    out = normalize_numpy(x, {"type": "none"})
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(np.where(invalid(x), 0, x)))


def test_scale_scalar_and_per_band():
    x = scene()
    check({"type": "scale", "value": 10000}, x, np.where(invalid(x), 0, x / 10000))
    values = [2.0, 4.0, 8.0]
    expected = np.where(invalid(x), 0, x / np.array(values, dtype=np.float32)[:, None, None])
    check({"type": "scale", "value": values}, x, expected)


def test_zscore_dataset():
    x = scene()
    mean, std = [100.0, 200.0, 300.0], [10.0, 20.0, 30.0]
    expected = (x - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    check(
        {"type": "zscore", "mean": mean, "std": std},
        x,
        np.where(invalid(x), 0, expected).astype(np.float32),
    )


def test_zscore_tile_ignores_nodata_and_nan():
    x = scene()
    check({"type": "zscore", "scope": "tile"}, x, zscore_ref(x))


def test_zscore_nan_nodata():
    x = scene(nodata=np.nan)
    check({"type": "zscore", "scope": "tile"}, x, zscore_ref(x, np.nan), nodata=np.nan)


def test_minmax_clips():
    x = scene()
    lo, hi = [100.0, 200.0, 300.0], [600.0, 700.0, 800.0]
    ref = (x - np.array(lo)[:, None, None]) / (np.array(hi) - np.array(lo))[:, None, None]
    check(
        {"type": "minmax", "min": lo, "max": hi},
        x,
        np.where(invalid(x), 0, np.clip(ref, 0, 1)).astype(np.float32),
    )
    check(
        {"type": "minmax", "min": lo, "max": hi, "clip": False},
        x,
        np.where(invalid(x), 0, ref).astype(np.float32),
    )


def test_percentile_tile():
    x = scene()
    expected = np.zeros_like(x)
    for band in range(3):
        values, keep = valid_values(x, band)
        lo, hi = np.percentile(values, [5, 95])
        expected[band][keep] = np.clip((values - lo) / (hi - lo), 0, 1)
    check({"type": "percentile", "low": 5, "high": 95, "scope": "tile"}, x, expected)


def test_db():
    x = scene()
    x[2, 15, 15] = 1e-12  # below eps
    expected = 10 * np.log10(np.maximum(x, 1e-6))
    check({"type": "db", "eps": 1e-6}, x, np.where(invalid(x), 0, expected).astype(np.float32))


def test_steps_apply_in_order():
    """db then zscore differs from zscore then db (not affine-invariant)."""
    x = scene()
    spec = [{"type": "db"}, {"type": "zscore", "scope": "tile"}]
    logged = np.where(invalid(x), x, 10 * np.log10(np.maximum(x, 1e-10)))
    check(spec, x, zscore_ref(logged))


def test_non_finite_intermediate_values_are_invalid():
    """log of a valid 0 is -inf; later statistics must ignore it, and it ends as 0."""
    x = np.array([[[0.0, 1.0, 10.0, 100.0]]], dtype=np.float32)
    out = normalize_numpy(
        x, [{"type": "db", "eps": 1e-30}, {"type": "zscore", "scope": "tile"}], nodata=None
    )
    assert np.isfinite(out).all()
    out = normalize_numpy(
        x, [lambda t, v: torch.log10(t), {"type": "zscore", "scope": "tile"}], nodata=None
    )
    expected = zscore_ref(
        np.log10(np.array([[[np.nan, 1.0, 10.0, 100.0]]], dtype=np.float32)), nodata=np.nan
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_large_uint16_values():
    x = np.array([[[0, 1, 40000, 65535]]], dtype=np.uint16)
    out = normalize_numpy(x, {"type": "scale", "value": 65535})
    np.testing.assert_allclose(out, [[[0, 1 / 65535, 40000 / 65535, 1]]], rtol=1e-6)


def test_callable_step():
    x = scene()

    def double(t, valid):
        return t * 2

    check(double, x, np.where(invalid(x), 0, 2 * x))


def test_vector_and_integer_inputs():
    """Integer input is converted to float32; nodata is still filled with 0."""
    x = (RNG.integers(1, 100, size=(2, 8, 8))).astype(np.uint16)
    x[:, 0, 0] = 0
    out = normalize_numpy(x, {"type": "scale", "value": 10})
    np.testing.assert_allclose(out, np.where(x == 0, 0, x / 10), rtol=1e-6)


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"type": "nope"}, "unknown normalisation type"),
        ({"value": 3}, "'type'"),
        ({"type": "scale"}, "needs 'value'"),
        ({"type": "zscore"}, "mean and std, or scope"),
        ({"type": "zscore", "scope": "global"}, "scope"),
        ({"type": "minmax", "min": [0]}, "needs 'max'"),
        ({"type": "percentile", "low": 98, "high": 2}, "low < high"),
        ({"type": "scale", "value": 1, "bogus": 2}, "unexpected keys"),
        (3, "dict"),
        ({"type": "scale", "value": 0}, "must not be 0"),
        ({"type": "zscore", "mean": [0], "std": [0]}, "positive"),
        ({"type": "zscore", "mean": [0], "std": [float("inf")]}, "finite"),
        ({"type": "db", "eps": 0}, "eps > 0"),
        ({"type": "minmax", "min": [1, 2], "max": [0, 3]}, "min < max"),
        ({"type": "minmax", "min": 0, "max": 1, "clip": "yes"}, "clip"),
        (
            [{"type": "zscore", "scope": "tile"}, {"type": "zscore", "scope": "scene"}],
            "can't follow",
        ),
        # a percentile without scope is per tile too
        ([{"type": "percentile"}, {"type": "zscore", "scope": "scene"}], "can't follow"),
    ],
)
def test_invalid_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        validate_spec(spec)


def test_per_band_length_must_match_bands():
    with pytest.raises(ValueError, match="3 bands"):
        normalize_numpy(scene(), {"type": "scale", "value": [1, 2]})


def test_scene_scope_is_an_error_in_batch_mode():
    with pytest.raises(ValueError, match="scene"):
        predict_batch(
            torch.nn.Identity(),
            scene()[None],
            progress=False,
            device="cpu",
            normalize={"type": "zscore", "scope": "scene"},
        )


@pytest.mark.parametrize(
    "step",
    [
        {"type": "zscore", "scope": "scene"},
        {"type": "percentile", "low": 2, "high": 98, "scope": "scene"},
    ],
)
def test_scene_scope_is_independent_of_tiling(step):
    """Scene statistics are computed once before tiling: an identity model returns the
    whole-scene normalised input whatever the patch size."""
    x = scene()
    x[1, 12, 12] = 5.0  # no NaN: the identity output would carry it
    expected = normalize_numpy(x, step)
    for patch, overlap in ((30, 0), (12, 4), (8, 3)):
        out = predict_array(
            torch.nn.Identity(),
            x,
            patch_size=patch,
            overlap=overlap,
            normalize=step,
            device="cpu",
            progress=False,
            mask_nodata=True,
        )
        keep = ~invalid(x).all(axis=0)
        np.testing.assert_allclose(out[:, keep], expected[:, keep], rtol=1e-5, atol=1e-5)


def test_tile_scope_depends_on_the_patch():
    x = scene()
    x[1, 12, 12] = 5.0
    step = {"type": "zscore", "scope": "tile"}
    whole = predict_array(
        torch.nn.Identity(),
        x,
        patch_size=20,
        overlap=0,
        normalize=step,
        device="cpu",
        progress=False,
    )
    tiled = predict_array(
        torch.nn.Identity(),
        x,
        patch_size=8,
        overlap=0,
        normalize=step,
        device="cpu",
        progress=False,
    )
    assert np.abs(whole - tiled).max() > 0.1


def test_step_dict_without_type_names_the_problem():
    with pytest.raises(ValueError, match="'type'"):
        predict_batch(
            torch.nn.Identity(), scene()[None], progress=False, device="cpu", normalize={"value": 2}
        )


def test_clip_step():
    x = scene()
    out = normalize_numpy(
        x, [{"type": "scale", "value": 1000}, {"type": "clip", "max": 0.5}], nodata=0
    )
    np.testing.assert_allclose(out, np.where(invalid(x), 0, np.minimum(x / 1000, 0.5)), rtol=1e-6)
    per_band = normalize_numpy(x, {"type": "clip", "min": [10, 20, 30], "max": 500}, nodata=0)
    expected = np.clip(x, np.array([10, 20, 30])[:, None, None], 500)
    np.testing.assert_allclose(per_band, np.where(invalid(x), 0, expected), rtol=1e-6)
    with pytest.raises(ValueError, match="'min', 'max'"):
        validate_spec({"type": "clip"})
    with pytest.raises(ValueError, match="min <= max"):
        validate_spec({"type": "clip", "min": 2, "max": 1})
