"""Dict inputs: rasters at other resolutions, time series, vectors."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manypred import predict_array, predict_batch

KW = {"device": "cpu", "progress": False, "nodata": None}
RNG = np.random.default_rng(0)


def scene(height=96, width=120):
    return {
        "s2": RNG.random((3, height, width)).astype(np.float32) + 0.1,
        "s1": RNG.random((2, height // 2, width // 2)).astype(np.float32) + 0.1,
    }


class Fusion(torch.nn.Module):
    """s2 (B, 3, H, W) + nearest-upsampled s1 (B, 2, H/2, W/2), pixel-local, so tiled must
    equal untiled exactly."""

    def forward(self, x):
        s1 = F.interpolate(x["s1"], scale_factor=2, mode="nearest")
        return x["s2"][:, :2] + s1


def untiled(model, inputs):
    batch = {k: torch.from_numpy(v)[None] for k, v in inputs.items()}
    with torch.no_grad():
        return model(batch)[0].numpy()


@pytest.mark.parametrize("patch, overlap", [(32, 0), (32, 8), (48, 16), (40, 12)])
def test_two_resolutions_tiled_equals_untiled(patch, overlap):
    inputs = scene()
    out = predict_array(Fusion(), inputs, patch_size=patch, overlap=overlap, **KW)
    np.testing.assert_allclose(out, untiled(Fusion(), inputs), rtol=1e-6)


def test_scene_not_multiple_of_the_scale():
    inputs = {
        "s2": RNG.random((3, 95, 121)).astype(np.float32) + 0.1,
        "s1": RNG.random((2, 48, 61)).astype(np.float32) + 0.1,
    }

    def model(x):
        s1 = F.interpolate(x["s1"], scale_factor=2, mode="nearest")
        return x["s2"][:, :2] + s1[:, :, : x["s2"].shape[2], : x["s2"].shape[3]]

    out = predict_array(model, inputs, patch_size=32, overlap=8, scales={"s1": 0.5}, **KW)
    expected = inputs["s2"][:2] + np.repeat(np.repeat(inputs["s1"], 2, 1), 2, 2)[:, :95, :121]
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_odd_patch_for_half_resolution_input_is_an_error():
    with pytest.raises(ValueError, match="divisible by 2"):
        predict_array(Fusion(), scene(), patch_size=33, overlap=8, **KW)


def test_scale_inferred_from_shape_must_be_consistent():
    inputs = scene()
    inputs["s1"] = inputs["s1"][:, :, :50]  # 48 x 50 for a 96 x 120 primary
    with pytest.raises(ValueError, match="s1"):
        predict_array(Fusion(), inputs, patch_size=32, overlap=8, **KW)


def test_vector_and_time_series_inputs():
    height, width, steps = 64, 80, 4
    inputs = {
        "s2": RNG.random((3, height, width)).astype(np.float32),
        "ts": RNG.random((steps, 3, height, width)).astype(np.float32),
        "ts_mask": np.array([True, True, False, True]),
        "ts_dates": np.array([10, 40, 70, 100], dtype=np.int64),
        "meta": np.array([0.5, -1.0, 2.0], dtype=np.float32),
    }
    seen = {}

    def model(x):
        seen["mask"], seen["dates"] = x["ts_mask"], x["ts_dates"]
        mask = x["ts_mask"][:, :, None, None, None].float()
        ts_mean = (x["ts"] * mask).sum(dim=1) / mask.sum(dim=1)
        return x["s2"] + ts_mean + x["meta"][:, :, None, None]

    out = predict_array(model, inputs, patch_size=32, overlap=8, **KW)
    keep = inputs["ts_mask"]
    expected = inputs["s2"] + inputs["ts"][keep].mean(axis=0) + inputs["meta"][:, None, None]
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    assert seen["mask"].dtype == torch.bool and seen["dates"].dtype == torch.int64
    assert seen["mask"].shape[1:] == (steps,)


def test_normalize_per_input_and_vectors_untouched():
    inputs = {
        "s2": RNG.random((3, 32, 32)).astype(np.float32) * 100,
        "meta": np.array([1000.0], dtype=np.float32),
    }
    seen = {}

    def model(x):
        seen["s2_max"], seen["meta"] = float(x["s2"].abs().max()), float(x["meta"][0, 0])
        return x["s2"]

    predict_array(
        model,
        inputs,
        patch_size=32,
        overlap=0,
        normalize={"s2": {"type": "scale", "value": 100}},
        **KW,
    )
    assert seen["s2_max"] <= 1 and seen["meta"] == 1000.0


def test_nodata_comes_from_the_primary_input():
    inputs = scene(64, 64)
    inputs["s2"][:, :16] = 0
    out = predict_array(
        Fusion(), inputs, patch_size=32, overlap=8, device="cpu", progress=False, nodata=0
    )
    assert np.isnan(out[:, :16]).all()
    assert np.all(out[:, 16:] > 0)


def test_batch_mode_dict_inputs():
    chips = [scene(32, 32) for _ in range(3)]
    out = predict_batch(Fusion(), chips, batch_size=2, **KW)
    for chip, pred in zip(chips, out, strict=True):
        np.testing.assert_allclose(pred, untiled(Fusion(), chip), rtol=1e-6)
    stacked = {name: np.stack([c[name] for c in chips]) for name in chips[0]}
    np.testing.assert_allclose(predict_batch(Fusion(), stacked, batch_size=2, **KW), out)


def test_manyseg_multi_input_model_runs():
    manyseg = pytest.importorskip("manyseg")
    torch.manual_seed(0)
    model = manyseg.create_model(
        inputs={
            "s2": {"in_channels": 3, "backbone": "resnet18"},
            "s1": {"in_channels": 2, "backbone": "resnet18", "alignment": "same_extent"},
        },
        pretrained=False,
        fusion={"type": "intermediate", "op": "concat"},
        decoder="unet",
        num_classes=4,
    ).eval()
    inputs = scene(150, 170)
    inputs["s1"] = RNG.random((2, 75, 85)).astype(np.float32)
    out = predict_array(
        model, inputs, patch_size=64, overlap=16, activation="softmax", output="argmax", **KW
    )
    assert out.shape == (1, 150, 170) and out.dtype == np.uint8
    chips = {"s2": inputs["s2"][None, :, :64, :64], "s1": inputs["s1"][None, :, :32, :32]}
    assert predict_batch(model, chips, **KW).shape == (1, 4, 64, 64)


def test_all_nodata_scene_with_reduced_output():
    image = np.zeros((3, 64, 64), dtype=np.float32)
    out = predict_array(
        lambda x: F.avg_pool2d(x, 4),
        image,
        patch_size=32,
        overlap=8,
        device="cpu",
        progress=False,
        nodata=0,
    )
    assert out.shape == (3, 16, 16) and np.isnan(out).all()


def test_non_integer_ratios_are_rejected():
    inputs = {"a": np.ones((1, 64, 64), np.float32), "b": np.ones((1, 96, 96), np.float32)}
    with pytest.raises(ValueError, match="whole numbers"):
        predict_array(lambda x: x["a"], inputs, patch_size=32, overlap=0, **KW)


def test_uint16_vectors_keep_large_values():
    seen = {}

    def model(x):
        seen["meta"] = x["meta"]
        return x["s2"]

    inputs = {
        "s2": np.ones((1, 32, 32), np.float32),
        "meta": np.array([40000, 65535], dtype=np.uint16),
    }
    predict_array(model, inputs, patch_size=32, overlap=0, **KW)
    assert seen["meta"].tolist() == [[40000, 65535]]


def test_scene_scope_normalisation_of_a_time_series():
    ts = RNG.random((2, 1, 32, 32)).astype(np.float32) * 10 + 5
    seen = {}

    def model(x):
        seen.setdefault("ts", x["ts"])  # the first patch, window (0, 16, 0, 16)
        return x["s2"]

    inputs = {"s2": np.ones((1, 32, 32), np.float32), "ts": ts}
    predict_array(
        model,
        inputs,
        patch_size=16,
        overlap=0,
        normalize={"ts": {"type": "zscore", "scope": "scene"}},
        **KW,
    )
    # pooled over both frames and the whole scene, so each patch is not zero-mean itself
    expected = (ts - ts.mean()) / ts.std()
    np.testing.assert_allclose(
        seen["ts"][0].numpy(), expected[:, :, :16, :16], rtol=1e-4, atol=1e-5
    )


def test_padding_value_per_input_dtype():
    """Edge windows past the scene pad each input with a value its dtype can hold."""
    inputs = {"s2": np.ones((1, 30, 30), np.int16), "mask": np.ones((1, 30, 30), np.uint8)}

    def model(x):
        return {"out": x["s2"] + x["mask"], "quarter": F.avg_pool2d(x["s2"], 4)}

    out = predict_array(
        model, inputs, patch_size=16, overlap=0, device="cpu", progress=False, nodata=-9999
    )
    assert out["out"].shape == (1, 30, 30)


def test_compile_warm_up_keeps_vector_dtypes(monkeypatch):
    dtypes = []

    def model(x):
        dtypes.append(x["ts_mask"].dtype)
        return x["s2"]

    monkeypatch.setattr(torch, "compile", lambda m, **kw: m)
    inputs = {"s2": np.ones((1, 32, 32), np.float32), "ts_mask": np.array([True, False])}
    predict_array(model, inputs, patch_size=32, overlap=0, compile=True, **KW)
    assert set(dtypes) == {torch.bool}


def test_ratio_inferred_when_sides_round_differently():
    """52 x 38 for a 103 x 77 primary: half resolution, one side rounded up, one down."""
    inputs = {
        "a": RNG.random((1, 103, 77)).astype(np.float32),
        "b": RNG.random((1, 52, 38)).astype(np.float32),
    }
    out = predict_array(lambda x: x["a"], inputs, patch_size=32, overlap=8, **KW)
    assert out.shape == (1, 103, 77)
    both_down = {
        "a": RNG.random((1, 13, 17)).astype(np.float32),
        "b": RNG.random((1, 6, 8)).astype(np.float32),
    }  # 6.5 -> 6, 8.5 -> 8
    from manypred.inputs import describe_inputs

    assert describe_inputs(both_down)[1]["b"].scale == 0.5


def test_ambiguous_rounded_ratio_needs_scales():
    """1 x 1 beside 2 x 3 fits both 1/2 and 1/3: rejected rather than guessed."""
    from manypred.inputs import describe_inputs

    inputs = {"a": np.ones((1, 2, 3), np.float32), "b": np.ones((1, 1, 1), np.float32)}
    with pytest.raises(ValueError):
        describe_inputs(inputs)


def test_time_series_alone_in_batch_mode_and_lazily():
    from manypred.lazy import ArrayReader

    ts = RNG.random((3, 2, 40, 48)).astype(np.float32)
    chips = [{"ts": ts[:, :, :32, :32]} for _ in range(2)]
    out = predict_batch(lambda x: x["ts"].mean(dim=1), chips, **KW)
    assert out.shape == (2, 2, 32, 32)
    eager = predict_array(lambda x: x["ts"].mean(dim=1), {"ts": ts}, patch_size=16, overlap=4, **KW)
    lazy = predict_array(
        lambda x: x["ts"].mean(dim=1), {"ts": ArrayReader(ts)}, patch_size=16, overlap=4, **KW
    )
    np.testing.assert_array_equal(lazy, eager)
