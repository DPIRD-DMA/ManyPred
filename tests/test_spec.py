"""tiling_spec() protocol."""

import json
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manypred import predict_array, predict_batch
from manypred.spec import get_tiling_spec, validate_tiling_spec

KW = {"device": "cpu", "progress": False, "nodata": None}
RNG = np.random.default_rng(0)

SPEC = {
    "inputs": {
        "s2": {"kind": "raster", "scale": 1.0},
        "s1": {"kind": "raster", "scale": 0.5},
        "meta": {"kind": "vector"},
    },
    "outputs": {
        "seg": {"scope": "pixel", "scale": 1.0, "kind": "segmentation"},
        "quarter": {"scope": "pixel", "scale": 0.25, "kind": "regression"},
        "flooded": {"scope": "image", "kind": "classification"},
    },
    "pad_multiple": 8,
    "max_size": None,
    "normalize": {"s2": {"type": "scale", "value": 10}, "s1": {"type": "scale", "value": 10}},
}


class SpecModel(torch.nn.Module):
    def __init__(self, spec=SPEC):
        super().__init__()
        self.spec = spec
        self.calls = []

    def tiling_spec(self):
        return self.spec

    def forward(self, x):
        self.calls.append(tuple(x["s2"].shape))
        s1 = F.interpolate(x["s1"], scale_factor=2, mode="nearest")
        seg = x["s2"][:, :2] + s1 + x["meta"][:, :1, None, None]
        return {"seg": seg, "quarter": F.avg_pool2d(seg, 4), "flooded": seg.mean(dim=(2, 3))}


def inputs(height=96, width=128):
    return {
        "s2": RNG.random((3, height, width)).astype(np.float32) * 10,
        "s1": RNG.random((2, height // 2, width // 2)).astype(np.float32) * 10,
        "meta": np.array([0.5], dtype=np.float32),
    }


def test_valid_spec_round_trips():
    assert validate_tiling_spec(SPEC) == SPEC


@pytest.mark.parametrize(
    "change, match",
    [
        ({"inputs": {"a": {"kind": "image"}}}, "kind"),
        ({"inputs": {"a": {"kind": "raster", "scale": 0}}}, "scale"),
        ({"outputs": {"a": {"scope": "tile"}}}, "scope"),
        ({"outputs": {"a": {"scope": "image", "scale": 0.5}}}, "image-level"),
        ({"outputs": {"a": {"scope": "pixel", "scale": 0.3}}}, "1 / f"),
        ({"outputs": {"a": {"scope": "pixel", "scale": 0.4}}}, "1 / f"),
        ({"outputs": []}, "must be a dict"),
        ({"inputs": 0}, "must be a dict"),
        ({"outputs": ""}, "must be a dict"),
        ({"pad_multiple": 0}, "pad_multiple"),
        ({"bogus": 1}, "unexpected"),
        ({"normalize": {"s2": {"type": "nope"}}}, "normalisation"),
    ],
)
def test_invalid_specs(change, match):
    with pytest.raises(ValueError, match=match):
        validate_tiling_spec({**SPEC, **change})


def test_spec_from_model_or_argument():
    assert get_tiling_spec([SpecModel()], None) == SPEC
    assert get_tiling_spec([torch.nn.Identity()], None) is None
    other = {**SPEC, "pad_multiple": 16}
    assert get_tiling_spec([SpecModel()], other)["pad_multiple"] == 16
    with pytest.raises(ValueError, match="ensemble"):
        get_tiling_spec([SpecModel(), SpecModel(other)], None)


def test_spec_fills_scales_normalisation_and_output_factors():
    model = SpecModel()
    data = inputs()
    out = predict_array(model, data, patch_size=32, overlap=8, output={"flooded": "logits"}, **KW)
    # normalisation from the spec: inputs scaled by 1/10
    expected = data["s2"][:2] / 10 + np.repeat(np.repeat(data["s1"] / 10, 2, 1), 2, 2) + 0.5
    np.testing.assert_allclose(out["seg"], expected, rtol=1e-5)
    # windows planned for the 1/4 output from the start: no re-planned first batch
    assert len(model.calls) == 4 * 5  # (96 - 8) / 24 -> 4 rows, (128 - 8) / 24 -> 5 columns


def test_explicit_arguments_win_over_the_spec():
    data = inputs()
    out = predict_array(
        SpecModel(),
        data,
        patch_size=32,
        overlap=8,
        normalize={},
        output={"flooded": "logits"},
        **KW,
    )
    expected = data["s2"][:2] + np.repeat(np.repeat(data["s1"], 2, 1), 2, 2) + 0.5
    np.testing.assert_allclose(out["seg"], expected, rtol=1e-5)


def test_patch_size_rounded_to_pad_multiple_and_capped_at_max_size():
    model = SpecModel({**SPEC, "max_size": 40})
    with pytest.warns(UserWarning, match="max_size"):
        predict_array(model, inputs(), patch_size=60, overlap=8, output={"flooded": "logits"}, **KW)
    assert {shape[-1] for shape in model.calls} == {40}
    model = SpecModel()
    predict_array(model, inputs(), patch_size=36, overlap=8, output={"flooded": "logits"}, **KW)
    assert {shape[-1] for shape in model.calls} == {32}


def test_batch_mode_uses_the_spec_normalisation():
    data = inputs(32, 32)
    chips = {name: x[None] for name, x in data.items()}
    out = predict_batch(SpecModel(), chips, output={"flooded": "logits"}, **KW)
    expected = data["s2"][:2] / 10 + np.repeat(np.repeat(data["s1"] / 10, 2, 1), 2, 2) + 0.5
    np.testing.assert_allclose(out["seg"][0], expected, rtol=1e-5)


def test_manyseg_tiling_spec_drives_tiling():
    """With ManySeg's tiling_spec() (ManySeg branch feat/tiling-spec), ManyPred plans windows
    for the 1/4 head from the start and infers the same_extent input's ratio from its shape."""
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
        heads={
            "seg": {"type": "segmentation", "out_channels": 3},
            "height": {"type": "regression", "out_channels": 1, "scale": 0.25},
        },
    ).eval()
    if not hasattr(model, "tiling_spec"):
        pytest.skip("this ManySeg has no tiling_spec()")
    spec = get_tiling_spec([model], None)
    assert spec["inputs"]["s1"]["scale"] is None and spec["pad_multiple"] == 32
    calls = []
    model.register_forward_hook(lambda m, i, o: calls.append(1))
    data = {
        "s2": RNG.random((3, 150, 170)).astype(np.float32),
        "s1": RNG.random((2, 75, 85)).astype(np.float32),
    }
    out = predict_array(
        model,
        data,
        patch_size=64,
        overlap=16,
        activation={"seg": "softmax"},
        output={"seg": "argmax"},
        batch_size=1,
        **KW,
    )
    assert out["seg"].shape == (1, 150, 170) and out["height"].shape == (1, 38, 43)
    windows = len(calls)
    # every call was a planned window: none redone for re-planning (3 x 4 grid of 64 / 48)
    assert windows == 3 * 4


def test_malformed_nested_specs_raise_value_errors():
    for bad in (
        {"inputs": {"x": None}},
        {"outputs": {"y": 3}},
        {"inputs": [1]},
        {"outputs": {"y": {"kind": 5}}},
    ):
        with pytest.raises(ValueError):
            validate_tiling_spec(bad)


def test_normalize_false_ignores_the_stored_normalisation():
    data = inputs()
    out = predict_array(
        SpecModel(),
        data,
        patch_size=32,
        overlap=8,
        normalize=False,
        output={"flooded": "logits"},
        **KW,
    )
    expected = data["s2"][:2] + np.repeat(np.repeat(data["s1"], 2, 1), 2, 2) + 0.5
    np.testing.assert_allclose(out["seg"], expected, rtol=1e-5)


def test_explicit_empty_scales_override_the_spec():
    """The spec says s1 is at half resolution; scales={} asks for shapes to decide."""
    spec = {**SPEC, "inputs": {**SPEC["inputs"], "s1": {"kind": "raster", "scale": 0.5}}}
    data = inputs()
    data["s1"] = np.repeat(np.repeat(data["s1"], 2, 1), 2, 2)  # now full resolution

    def model(x):
        return {"seg": x["s2"][:, :2] + x["s1"], "flooded": x["s2"].mean(dim=(2, 3))}

    spec["outputs"] = {k: v for k, v in SPEC["outputs"].items() if k != "quarter"}
    with pytest.raises(ValueError, match="s1"):
        predict_array(
            model, data, patch_size=32, overlap=8, spec=spec, output={"flooded": "logits"}, **KW
        )
    out = predict_array(
        model,
        data,
        patch_size=32,
        overlap=8,
        spec=spec,
        scales={},
        output={"flooded": "logits"},
        **KW,
    )
    assert out["seg"].shape == (2, 96, 128)


class _NamedSingle(torch.nn.Module):
    """Returns a bare tensor; its tiling_spec names the output "mask"."""

    def forward(self, x):
        return x

    def tiling_spec(self):
        return {"outputs": {"mask": {"scope": "pixel", "kind": "segmentation"}}}


def test_single_output_model_settings_by_head_name():
    """A single-output model returns a bare tensor; settings keyed by its head name apply, and
    a key that is not its name is an error."""
    out = predict_array(
        _NamedSingle(),
        np.ones((2, 32, 32), np.float32),
        patch_size=32,
        overlap=0,
        activation={"mask": "softmax"},
        output={"mask": "argmax"},
        **KW,
    )
    assert out.dtype == np.uint8
    with pytest.raises(ValueError, match="model's output is 'mask'"):
        predict_array(
            _NamedSingle(),
            np.ones((2, 32, 32), np.float32),
            patch_size=32,
            overlap=0,
            output={"wrong": "argmax"},
            **KW,
        )


def test_a_lone_time_series_is_the_primary_input():
    ts = RNG.random((3, 2, 40, 48)).astype(np.float32)
    ts[:, :, :8] = 0  # nodata in every frame and band

    def model(x):
        return x["ts"].mean(dim=1)

    out = predict_array(
        model, {"ts": ts}, patch_size=16, overlap=4, device="cpu", progress=False, nodata=0
    )
    np.testing.assert_allclose(out[:, 8:], ts.mean(axis=0)[:, 8:], rtol=1e-5)
    assert np.isnan(out[:, :8]).all()


def test_spec_activations_are_defaults():
    spec = {
        "outputs": {
            "seg": {"scope": "pixel", "activation": "softmax"},
            "score": {"scope": "image", "activation": "none"},
        }
    }

    def model(x):
        return {"seg": x[:, :2], "score": x.mean(dim=(2, 3))}

    image = RNG.random((3, 32, 32)).astype(np.float32)
    out = predict_array(model, image, patch_size=32, overlap=0, spec=spec, **KW)
    np.testing.assert_allclose(out["seg"].sum(axis=0), 1, rtol=1e-5)  # softmax, probs
    assert out["score"].values.shape == (1, 3)  # "none" counts as an explicit choice
    raw = predict_array(
        model, image, patch_size=32, overlap=0, spec=spec, activation={"seg": "none"}, **KW
    )
    np.testing.assert_allclose(raw["seg"], image[:2], rtol=1e-6)  # explicit "none" wins
    classes = predict_array(
        model, image, patch_size=32, overlap=0, spec=spec, output={"seg": "argmax"}, **KW
    )
    assert classes["seg"].dtype == np.uint8
    with pytest.raises(ValueError, match="activation"):
        validate_tiling_spec({"outputs": {"a": {"activation": "relu"}}})


def test_manyseg_saved_model_needs_no_settings(tmp_path):
    """A ManySeg model saved with its normalisation and activations predicts correctly with
    no normalize= or activation= given."""
    manyseg = pytest.importorskip("manyseg")
    if not hasattr(manyseg, "save"):
        pytest.skip("this ManySeg has no save/load")
    torch.manual_seed(0)
    model = manyseg.create_model(
        backbone="resnet18",
        pretrained=False,
        in_channels=3,
        heads={
            "seg": {"type": "segmentation", "out_channels": 3},
            "height": {"type": "regression", "out_channels": 1, "scale": 0.25},
        },
    )
    norm = [{"type": "scale", "value": 10000}, {"type": "zscore", "scope": "tile"}]
    path = manyseg.save(
        model, tmp_path / "m.safetensors", normalize=norm, activations={"seg": "softmax"}
    )
    loaded = manyseg.load(path)
    image = (RNG.random((3, 96, 96)) * 10000).astype(np.uint16) + 1
    out = predict_array(loaded, image, patch_size=96, overlap=0, **KW)
    np.testing.assert_allclose(out["seg"].sum(axis=0), 1, rtol=1e-5)
    # the same as normalising and activating by hand on the whole (single-patch) scene
    from manypred import normalize_torch

    x = normalize_torch(torch.from_numpy(image.astype(np.float32))[None], norm)
    with torch.no_grad():
        expected = torch.softmax(loaded(x)["seg"], dim=1)[0].numpy()
    np.testing.assert_allclose(out["seg"], expected, atol=1e-5)
    assert out["height"].shape == (1, 24, 24)


def test_explicit_settings_win_over_spec_activations():
    spec = {
        "outputs": {
            "seg": {"scope": "pixel", "activation": "softmax"},
            "aux": {"scope": "pixel", "activation": "sigmoid"},
        }
    }

    def model(x):
        return {"seg": x[:, :2], "aux": x[:, 2:]}

    image = RNG.random((3, 32, 32)).astype(np.float32)
    kw = {"patch_size": 32, "overlap": 0, "spec": spec, **KW}
    raw = predict_array(model, image, output="logits", **kw)  # explicit logits: no activation
    np.testing.assert_allclose(raw["seg"], image[:2], rtol=1e-6)
    np.testing.assert_allclose(raw["aux"], image[2:], rtol=1e-6)
    partial = predict_array(model, image, activation={"aux": "none"}, **kw)
    np.testing.assert_allclose(partial["seg"].sum(axis=0), 1, rtol=1e-5)  # spec default
    np.testing.assert_allclose(partial["aux"], image[2:], rtol=1e-6)  # explicit override
    stored = {
        "outputs": {"seg": {"activation": "softmax"}},
        "normalize": [{"type": "scale", "value": 1000}],
    }
    unnormalised = predict_array(
        lambda x: x[:, :2],
        image,
        output="logits",
        spec=stored,
        normalize=False,
        patch_size=32,
        overlap=0,
        **KW,
    )
    np.testing.assert_allclose(unnormalised, image[:2], rtol=1e-6)


def test_single_output_model_uses_the_spec_activation():
    spec = {"outputs": {"logits": {"scope": "pixel", "activation": "softmax"}}}
    image = RNG.random((3, 32, 32)).astype(np.float32)
    probs = predict_array(lambda x: x, image, patch_size=32, overlap=0, spec=spec, **KW)
    np.testing.assert_allclose(probs.sum(axis=0), 1, rtol=1e-5)
    chips = predict_batch(lambda x: x, image[None], spec=spec, **KW)
    np.testing.assert_allclose(chips.sum(axis=1), 1, rtol=1e-5)


def test_spec_version():
    """Absent means 1; a newer version is refused with a clear message, not misread."""
    base = {"outputs": {"mask": {"scope": "pixel"}}}
    validate_tiling_spec(base)
    validate_tiling_spec({**base, "version": 1})
    with pytest.raises(ValueError, match="version 2; this ManyPred reads up to 1"):
        validate_tiling_spec({**base, "version": 2})
    for bad in (0, True, "1"):
        with pytest.raises(ValueError, match="positive integer"):
            validate_tiling_spec({**base, "version": bad})


# --- what a saved ManySeg model hands over ---------------------------------------------------


class _Declares(torch.nn.Module):
    """Two rasters (or a raster and a vector) and whatever spec a test gives it. Each output is
    the matching input's mean, so a test can see what normalisation reached it."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def tiling_spec(self):
        return self.spec

    def forward(self, x):
        out = x["a"][:, :1]
        return {"o": out, "b_mean": x["b"].mean(dim=(1, 2, 3))[:, None, None, None] + 0 * out}


def _two_rasters():
    return {
        "a": np.full((1, 32, 32), 100.0, dtype=np.float32),
        "b": np.full((1, 32, 32), 7.0, dtype=np.float32),
    }


TINY = {"patch_size": 32, "overlap": 0, "device": "cpu", "dtype": "float32", "progress": False}


@pytest.mark.parametrize("given", [False, {"a": {"type": "scale", "value": 10}}])
def test_explicit_normalize_replaces_a_bad_stored_one(given):
    bad = {"inputs": {"a": {}, "b": {}}, "normalize": [{"type": "zscor"}]}
    model = _Declares(bad)
    with pytest.raises(ValueError, match="zscor"):
        predict_array(model, _two_rasters(), **TINY, output="logits")
    out = predict_array(model, _two_rasters(), **TINY, output="logits", normalize=given)
    assert out["o"][0, 0, 0] == (100 if given is False else 10)
    with pytest.raises(ValueError, match="zscor"):  # spec= is the caller's own: still checked
        predict_array(model, _two_rasters(), **TINY, output="logits", spec=bad, normalize=False)


def test_flat_normalize_goes_to_the_only_raster():
    class WithVector(torch.nn.Module):
        def tiling_spec(self):
            return {
                "inputs": {"img": {"kind": "raster", "scale": 1.0}, "doy": {"kind": "vector"}},
                "outputs": {"o": {"scope": "pixel", "scale": 1.0, "activation": "none"}},
                "normalize": [{"type": "scale", "value": 10}],
            }

        def forward(self, x):
            return {"o": x["img"] + x["doy"][:, :, None, None]}

    scene = {
        "img": np.full((1, 32, 32), 100.0, dtype=np.float32),
        "doy": np.array([1.0], dtype=np.float32),
    }
    assert predict_array(WithVector(), scene, **TINY)["o"][0, 0, 0] == 11  # 100 / 10 + 1
    chips = {"img": scene["img"][None], "doy": scene["doy"][None]}
    out = predict_batch(WithVector(), chips, device="cpu", dtype="float32", progress=False)
    assert out["o"][0, 0, 0, 0] == 11


def test_flat_normalize_is_still_ambiguous_with_two_rasters():
    model = _Declares({"inputs": {"a": {}, "b": {}}, "normalize": [{"type": "scale", "value": 2}]})
    with pytest.raises(ValueError, match="several inputs"):
        predict_array(model, _two_rasters(), **TINY, output="logits")


def test_primary_input_is_the_specs_first_raster_whatever_the_dict_order():
    """The model's first input sets the grid and the nodata mask."""
    model = _Declares({"inputs": {"a": {"kind": "raster"}, "b": {"kind": "raster"}}})
    a = np.full((1, 32, 32), 100.0, dtype=np.float32)
    a[:, :16] = 0  # nodata in the primary input only
    b = np.full((1, 32, 32), 7.0, dtype=np.float32)
    keep = {**TINY, "output": "logits", "nodata": 0, "normalize": False, "shift_patches": False}
    out = predict_array(model, {"b": b, "a": a}, **keep)["o"]
    assert np.isnan(out[0, :16]).all() and (out[0, 16:] == 100).all()
    # reversed and of different sizes: planned on "a", so the coarser "b" is its half
    half = {"inputs": {"a": {"kind": "raster"}, "b": {"kind": "raster", "scale": 0.5}}}
    small = np.full((1, 16, 16), 7.0, dtype=np.float32)
    tiled = predict_array(_Declares(half), {"b": small, "a": a}, **keep)["o"]
    assert tiled.shape == (1, 32, 32)
    # batch mode too: the valid mask comes from "a"
    chips = {"b": b[None], "a": a[None]}
    batch = predict_batch(
        model, chips, device="cpu", dtype="float32", progress=False,
        output="logits", nodata=0, normalize=False,
    )["o"]  # fmt: skip
    assert np.isnan(batch[0, 0, :16]).all() and (batch[0, 0, 16:] == 100).all()


def test_probs_keeps_an_output_declared_as_already_the_prediction():
    class TwoHeads(torch.nn.Module):
        def tiling_spec(self):
            return {
                "outputs": {
                    "seg": {"scope": "pixel", "scale": 1.0, "activation": "softmax"},
                    "height": {"scope": "pixel", "scale": 1.0, "activation": "none"},
                }
            }

        def forward(self, x):
            return {"seg": torch.cat([x, -x], dim=1), "height": x * 3}

    scene = np.ones((1, 32, 32), dtype=np.float32)
    out = predict_array(TwoHeads(), scene, **TINY, output="probs", normalize=False)
    assert out["height"][0, 0, 0] == 3
    assert np.allclose(out["seg"][:, 0, 0].sum(), 1)
    # asking for both at once on an output the caller set to "none" is still a contradiction
    with pytest.raises(ValueError, match="needs an activation"):
        predict_array(
            TwoHeads(), scene, **TINY, output="probs", normalize=False,
            activation={"seg": "softmax", "height": "none"},
        )  # fmt: skip


def test_manyseg_flat_spec_beside_a_vector(tmp_path):
    """A flat normalisation saved on one raster and a vector, as ManySeg's own files are."""
    manyseg = pytest.importorskip("manyseg")
    if not hasattr(manyseg, "save"):
        pytest.skip("this ManySeg has no save/load")
    model = manyseg.create_model(
        backbone="resnet18",
        pretrained=False,
        inputs={
            "img": {"in_channels": 3},
            "doy": {"kind": "vector", "fields": {"doy": {"type": "cyclic", "period": 365.25}}},
        },
        heads={
            "seg": {"type": "segmentation", "out_channels": 2},
            "height": {"type": "regression", "out_channels": 1},
        },
    )
    path = manyseg.save(
        model,
        tmp_path / "m.safetensors",
        normalize=[{"type": "scale", "value": 10000}],
        activations={"seg": "softmax", "height": "none"},
    )
    loaded = manyseg.load(path)
    scene = {
        "doy": np.array([100.0], dtype=np.float32),
        "img": np.random.default_rng(0).integers(1, 9000, (3, 64, 64)).astype(np.float32),
    }
    kw = {"patch_size": 64, "overlap": 0, "device": "cpu", "dtype": "float32", "progress": False}
    out = predict_array(loaded, scene, **kw, output="probs")
    assert out["seg"].shape == (2, 64, 64) and out["height"].shape == (1, 64, 64)
    x = {
        "img": torch.from_numpy(scene["img"] / 10000)[None],
        "doy": torch.from_numpy(scene["doy"])[None],
    }
    with torch.no_grad():
        raw = loaded.eval()(x)
    assert np.allclose(out["height"], raw["height"][0].numpy(), atol=1e-5)
    assert np.allclose(out["seg"], torch.softmax(raw["seg"], 1)[0].numpy(), atol=1e-5)


@pytest.mark.parametrize("empty", [[], {"type": "none"}, [{"type": "none"}]])
def test_a_spec_with_no_steps_is_a_no_op_for_any_number_of_rasters(empty):
    for model, kw in (
        (_Declares({"inputs": {"a": {}, "b": {}}}), {"normalize": empty}),
        (_Declares({"inputs": {"a": {}, "b": {}}, "normalize": empty}), {}),
    ):
        out = predict_array(model, _two_rasters(), **TINY, output="logits", **kw)
        assert out["o"][0, 0, 0] == 100
        chips = {name: x[None] for name, x in _two_rasters().items()}
        batch = predict_batch(
            model, chips, device="cpu", dtype="float32", progress=False, output="logits", **kw
        )
        assert batch["o"][0, 0, 0, 0] == 100


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("compile", [False, True])
def test_float_dates_keep_their_values_under_16_bit_dtypes(dtype, compile):
    if dtype == "float16" and compile:
        pytest.skip("one 16-bit dtype is enough to compile on the CPU")
    seen = {}

    class Series(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            seen["dates"] = x["ts_dates"]
            seen["series"] = x["ts"].dtype
            seen["meta"] = x["meta"].dtype
            return x["ts"].mean(dim=1)[:, :1] * self.scale

    scene = {
        "ts": np.ones((3, 1, 32, 32), dtype=np.float32),
        "ts_dates": np.array([18000.0, 18016.0, 18050.0], dtype=np.float32),
        "meta": np.array([1.0], dtype=np.float32),
    }
    kw = {**TINY, "dtype": dtype, "compile": compile, "output": "logits", "normalize": False}
    predict_array(Series(), scene, **kw)
    assert seen["dates"].dtype == torch.float32
    assert seen["dates"].tolist() == [[18000.0, 18016.0, 18050.0]]
    assert seen["series"] == seen["meta"] == getattr(torch, dtype)


class _Regression(torch.nn.Module):
    """A class head and a regression head, whose channels aren't classes."""

    def __init__(self, **extra):
        super().__init__()
        self.extra = extra

    def tiling_spec(self):
        return {
            "outputs": {
                "seg": {"scope": "pixel", "kind": "segmentation", "activation": "softmax"},
                "height": {"scope": "pixel", "kind": "regression", "activation": "none"},
            },
            **self.extra,
        }

    def forward(self, x):
        return {"seg": torch.cat([x, -x], dim=1), "height": torch.cat([x * 3, x * 5], dim=1)}


def test_global_argmax_leaves_regression_outputs_their_values():
    scene = np.ones((1, 32, 32), dtype=np.float32)
    keep = {**TINY, "normalize": False}
    with pytest.warns(UserWarning, match="regression output 'height'") as record:
        out = predict_array(_Regression(), scene, **keep, output="argmax")
    assert {w.filename for w in record if "regression" in str(w.message)} == {__file__}
    assert out["seg"].dtype == np.uint8 and out["seg"].shape == (1, 32, 32)
    assert out["height"].dtype == np.float32 and out["height"][1, 0, 0] == 5
    chips = scene[None]
    with pytest.warns(UserWarning, match="regression output 'height'"):
        batch = predict_batch(
            _Regression(), chips, device="cpu", dtype="float32", progress=False,
            normalize=False, output="argmax",
        )  # fmt: skip
    assert batch["height"][0, 1, 0, 0] == 5
    # asked for by name, it is what the caller wants, with no warning
    named = predict_array(
        _Regression(), scene, **keep, output={"seg": "argmax", "height": "argmax"}
    )
    assert named["height"].dtype == np.uint8


def test_a_lone_regression_output_keeps_its_values_too():
    spec = {"outputs": {"h": {"scope": "pixel", "kind": "regression", "activation": "none"}}}
    scene = np.ones((2, 32, 32), dtype=np.float32)
    with pytest.warns(UserWarning, match="regression output of this model"):
        out = predict_array(lambda x: x * 2, scene, **TINY, spec=spec, output="argmax")
    assert out.dtype == np.float32 and out[0, 0, 0] == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"bands": "red"},
        {"bands": [1, 2]},
        {"bands": {"a": "red"}},
        {"outputs": {"o": {"classes": "water"}}},
        {"outputs": {"o": {"classes": [0, 1]}}},
    ],
)
def test_names_in_a_spec_are_lists_of_strings(bad):
    with pytest.raises(ValueError, match="list of names"):
        validate_tiling_spec(bad)


NAMED = {
    "outputs": {
        "seg": {"scope": "pixel", "kind": "segmentation", "activation": "softmax",
                "classes": ["water", "land"]},
        "height": {"scope": "pixel", "kind": "regression", "activation": "none"},
    },
    "bands": ["red", "green"],
}  # fmt: skip


def _files_model():
    class Named(torch.nn.Module):
        def forward(self, x):
            return {"seg": x, "height": x[:, :1] * 3}

    return Named()


def _write_scene(path):
    rio = pytest.importorskip("rasterio")
    with rio.open(path, "w", driver="GTiff", height=32, width=32, count=2, dtype="float32") as dst:
        dst.write(np.ones((2, 32, 32), dtype=np.float32))
    return rio


def _load(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read().astype(np.float32), src.profile


@pytest.mark.parametrize("lazy", [False, True])
def test_predict_files_names_classes_and_bands(tmp_path, lazy):
    rio = _write_scene(tmp_path / "s.tif")
    from manypred import predict_files
    from manypred.geo import RasterioReader

    def load(path):
        return (RasterioReader(path), rio.open(path).profile) if lazy else _load(path)

    kw = {"patch_size": 32, "overlap": 0, "device": "cpu", "dtype": "float32", "progress": False}
    for mode, suffix in (("logits", "logits"), ("argmax", "argmax")):
        out = tmp_path / suffix
        out.mkdir()
        results = predict_files(
            _files_model(), [tmp_path / "s.tif"], load_func=load, output_dir=out, spec=NAMED,
            normalize=False, output={"seg": mode, "height": "logits"}, **kw,
        )  # fmt: skip
        manifest = json.loads(results[0].read_text())
        assert manifest["classes"] == {"seg": ["water", "land"]}
        assert manifest["input_bands"] == ["red", "green"]
        with rio.open(out / manifest["files"]["seg"]) as src:
            assert json.loads(src.tags()["input_bands"]) == ["red", "green"]
            if mode == "logits":
                assert src.descriptions == ("water", "land")
            else:
                assert json.loads(src.tags(1)["classes"]) == ["water", "land"]
                assert src.descriptions == (None,)
        with rio.open(out / manifest["files"]["height"]) as src:
            assert src.descriptions == (None,) and "classes" not in src.tags(1)


def test_predict_files_doesnt_name_what_postprocess_or_a_callable_made(tmp_path):
    rio = _write_scene(tmp_path / "s.tif")
    from manypred import predict_files

    kw = {"patch_size": 32, "overlap": 0, "device": "cpu", "dtype": "float32", "progress": False}
    made = {
        "postprocess": {"postprocess": lambda r: r, "output": "logits"},
        "callable": {"output": {"seg": lambda x: x, "height": "logits"}},
    }
    for name, extra in made.items():
        out = tmp_path / name
        out.mkdir()
        results = predict_files(
            _files_model(), [tmp_path / "s.tif"], load_func=_load, output_dir=out, spec=NAMED,
            normalize=False, **kw, **extra,
        )  # fmt: skip
        manifest = json.loads(results[0].read_text())
        assert "classes" not in manifest
        with rio.open(out / manifest["files"]["seg"]) as src:
            assert src.descriptions == (None, None)


def test_a_spec_without_names_writes_no_names(tmp_path):
    rio = _write_scene(tmp_path / "s.tif")
    from manypred import predict_files

    kw = {"patch_size": 32, "overlap": 0, "device": "cpu", "dtype": "float32", "progress": False}
    results = predict_files(
        _files_model(), [tmp_path / "s.tif"], load_func=_load, output_dir=tmp_path, normalize=False,
        output="logits", **kw,
    )  # fmt: skip
    manifest = json.loads(results[0].read_text())
    assert set(manifest) == {"files", "values"}
    with rio.open(tmp_path / manifest["files"]["seg"]) as src:
        assert "input_bands" not in src.tags()


@pytest.mark.parametrize(
    ("dtype", "classes", "description", "tag"),
    [
        ("float32", ["bg", "water"], None, None),  # one probability band: not one band per class
        ("uint8", ["bg", "water"], None, ["bg", "water"]),  # class indices: 0 is bg, 1 is water
        ("uint8", ["water"], "water", None),  # one name, one band: a sigmoid head's 0/1 band
    ],
)
def test_one_band_with_one_or_two_class_names(tmp_path, dtype, classes, description, tag):
    rio = pytest.importorskip("rasterio")
    from manypred.geo import write_geotiff

    path = write_geotiff(
        tmp_path / "o.tif", np.zeros((1, 8, 8), dtype=dtype), classes=classes, input_bands=None
    )
    with rio.open(path) as src:
        assert src.descriptions == (description,)
        assert (json.loads(src.tags(1)["classes"]) if tag else None) == tag
        assert tag is not None or "classes" not in src.tags(1)


def test_float_dates_are_float32_when_probing_an_all_nodata_scene():
    """A scene without a patch to predict is probed once, and gets what a batch would."""
    seen = {}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            seen["dates"] = x["ts_dates"].dtype
            return x["ts"].mean(dim=1) * self.weight

    inputs = {
        "ts": np.zeros((3, 1, 32, 32), np.float32),
        "ts_dates": np.array([1.0, 2.0, 3.0], np.float16),
    }
    predict_array(
        Model(),
        inputs,
        patch_size=32,
        overlap=0,
        **{**KW, "nodata": 0},
    )
    assert seen["dates"] == torch.float32


def test_regression_warning_on_an_all_nodata_scene():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            return {"height": x * self.weight, "seg": torch.cat([x, x], 1) * self.weight}

        def tiling_spec(self):
            return {"outputs": {"height": {"kind": "regression"}, "seg": {"kind": "segmentation"}}}

    with pytest.warns(UserWarning, match="regression output 'height'"):
        predict_array(
            Model(),
            np.zeros((1, 32, 32), np.float32),
            patch_size=32,
            overlap=0,
            output="argmax",
            **{**KW, "nodata": 0},
        )


def test_class_names_dropped_for_a_callable_output_given_by_the_empty_name(tmp_path):
    pytest.importorskip("rasterio")
    from manypred import predict_files

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            return torch.cat([x, x], 1) * self.weight

        def tiling_spec(self):
            return {"outputs": {"seg": {"classes": ["water", "land"]}}}

    written = predict_files(
        Model(),
        [tmp_path / "scene.tif"],
        load_func=lambda item: np.ones((1, 32, 32), np.float32),
        output={"": lambda values: values.flip(1)},
        output_dir=tmp_path / "out",
        patch_size=32,
        overlap=0,
        **KW,
    )
    import rasterio

    with rasterio.open(written[0]) as dataset:
        assert dataset.descriptions == (None, None)


@pytest.mark.parametrize(
    "scale, factor",
    [
        (0.01, 100),
        (1 / 3, 3),
        (0.25, 4),
        (1 / 96, 96),
        (0.33333333, 3),
        (0.01666667, 60),
        ("1/3", 3),
    ],
)
def test_output_scales_are_exact(scale, factor):
    """Scales were rounded to a denominator of at most 64, so 1/100 silently became 1/64."""
    from manypred.spec import named_output_factors, output_factors

    spec = validate_tiling_spec({"outputs": {"a": {"scale": scale}}})
    assert output_factors(spec) == [factor]
    assert named_output_factors(spec) == {"a": factor}


def test_single_output_model_merge_by_head_name():
    """merge, like output and activation, takes the spec's name for a bare tensor's head."""
    image = RNG.random((2, 48, 48)).astype(np.float32)
    cut = predict_array(
        _NamedSingle(), image, patch_size=32, overlap=8, merge={"mask": "cut"}, **KW
    )
    same = predict_array(_NamedSingle(), image, patch_size=32, overlap=8, merge="cut", **KW)
    np.testing.assert_array_equal(cut, same)
    with pytest.raises(ValueError, match="model's output is 'mask'"):
        predict_array(_NamedSingle(), image, patch_size=32, overlap=8, merge={"wrong": "cut"}, **KW)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.5, Fraction(1, 2)),
        (2, Fraction(2)),
        (0.16666667, Fraction(1, 6)),
        ("1/4", Fraction(1, 4)),
        (2 / 3, Fraction(2, 3)),
        (0.0137, Fraction(1, 73)),
        (0.333, Fraction(1, 3)),
        (0.0167, Fraction(1, 60)),
        (0.3, None),
        (0.75, None),
        (1.5, None),
        (1e-309, None),
        (Fraction(2, 99), Fraction(1, 50)),
    ],
)
def test_ratio(value, expected):
    """Decimals within 2% of k or 1 / k snap to it; other ratios stay other (and are
    refused)."""
    from manypred.inputs import ratio

    got = ratio(value)
    if expected is None:
        assert got.numerator != 1 and got.denominator != 1
    else:
        assert got == expected


def test_ratio_of_a_bad_string_is_a_value_error():
    from manypred.inputs import ratio

    with pytest.raises(ValueError):
        ratio("1/0")
    with pytest.raises(ValueError, match="scale"):
        validate_tiling_spec({"outputs": {"a": {"scale": "1/0"}}})
