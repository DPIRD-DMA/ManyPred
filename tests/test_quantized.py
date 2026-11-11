"""Models with int8 weights in torchao tensors, as ``manyseg.quantize.quantize_weights`` makes
them, and saved ``quantize_pt2e`` programs. Skipped without torchao; the tests on real ManySeg
models also need ManySeg."""

import numpy as np
import pytest
import torch

from manypred import predict_array

pytest.importorskip("torchao")
from torchao.quantization import IntxWeightOnlyConfig, quantize_  # noqa: E402
from torchao.quantization.granularity import PerAxis  # noqa: E402

F16, BF16, F32 = torch.float16, torch.bfloat16, torch.float32
KW = {"patch_size": 64, "overlap": 16, "progress": False}
MODES = ["int8_weight_only", "int8_dynamic"]


def scene(seed=0):
    return np.random.default_rng(seed).random((3, 96, 112)).astype(np.float32)


def int8(model):
    """Int8 conv weights, with the settings ``quantize_weights`` uses for convs."""
    config = IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0))
    quantize_(model, config, filter_fn=lambda m, _: isinstance(m, torch.nn.Conv2d))
    return model


def assert_int8_in_float32(weight):
    assert type(weight) not in (torch.Tensor, torch.nn.Parameter)
    assert weight.dtype == F32 and weight.scale.dtype == F32 and weight.qdata.dtype == torch.int8


@pytest.fixture
def manyseg_models():
    """make(mode): a small ManySeg model through ``quantize_weights``."""
    manyseg = pytest.importorskip("manyseg")
    quantize = pytest.importorskip("manyseg.quantize")

    def float_model():
        torch.manual_seed(0)
        return manyseg.create_model(
            backbone="resnet18",
            pretrained=False,
            in_channels=3,
            heads={"seg": {"type": "segmentation", "out_channels": 3}},
        ).eval()

    def make(mode="int8_weight_only"):
        return quantize.quantize_weights(float_model(), mode)

    make.float_model = float_model
    return make


@pytest.fixture
def saved_program(manyseg_models, tmp_path):
    """make(**quantize_pt2e_kwargs): a ``quantize_pt2e`` model saved and loaded by ManySeg."""
    manyseg = pytest.importorskip("manyseg")
    quantize = pytest.importorskip("manyseg.quantize")
    if not hasattr(quantize, "QuantizedProgram"):
        pytest.skip("needs a ManySeg that saves quantised programs")

    def make(batch=1, size=64, map_location="cpu", **kwargs):
        model = manyseg_models.float_model()
        calibration = [torch.rand(batch, 3, size, size) for _ in range(3)]
        program = quantize.quantize_pt2e(
            model, torch.rand(batch, 3, size, size), calibration, **kwargs
        )
        path = tmp_path / f"model_{batch}_{size}_{len(kwargs)}.pt2"
        manyseg.save(program, path, source=model, activations={"seg": "softmax"})
        return manyseg.load(path, map_location=map_location)

    return make


def close_to_float(out, manyseg_models):
    """Int8 changes values a little; the point here is that the patches lined up."""
    expected = predict_array(
        manyseg_models.float_model(),
        scene(),
        device="cpu",
        dtype="float32",
        activation="softmax",
        **KW,
    )
    assert out["seg"].shape == expected["seg"].shape
    assert np.isfinite(out["seg"]).all()
    assert (out["seg"].argmax(0) == expected["seg"].argmax(0)).mean() > 0.8


def test_saved_program_with_the_defaults(saved_program, manyseg_models):
    program = saved_program()
    assert program.batch_size == 1 and program.fixed_size is None
    close_to_float(predict_array(program, scene(), output="probs", **KW), manyseg_models)


def test_saved_program_reads_its_spec(saved_program):
    """The activation comes from the file: probabilities without activation=."""
    out = predict_array(saved_program(), scene(), output="probs", **KW)
    np.testing.assert_allclose(out["seg"].sum(0), 1, atol=1e-3)


def test_saved_program_with_a_fixed_batch(saved_program, manyseg_models):
    program = saved_program(batch=3, size=64)
    assert program.batch_size == 3
    close_to_float(predict_array(program, scene(), output="probs", **KW), manyseg_models)


def test_saved_program_with_a_dynamic_batch(saved_program, manyseg_models):
    program = saved_program(batch=2, dynamic_batch=True)
    assert program.batch_size is None
    out = predict_array(program, scene(), batch_size=2, output="probs", **KW)
    close_to_float(out, manyseg_models)


def test_saved_program_with_a_fixed_size(saved_program, manyseg_models):
    program = saved_program(size=64, dynamic=False)
    assert tuple(program.fixed_size) == (64, 64)
    with pytest.warns(UserWarning, match="using 64"):
        out = predict_array(
            program, scene(), patch_size=1000, overlap=16, output="probs", progress=False
        )
    close_to_float(out, manyseg_models)


def save_and_load(model, example, tmp_path, name, activations):
    manyseg = pytest.importorskip("manyseg")
    quantize = pytest.importorskip("manyseg.quantize")
    if not hasattr(quantize, "QuantizedProgram"):
        pytest.skip("needs a ManySeg that saves quantised programs")
    make = example if callable(example) else (lambda: example)
    program = quantize.quantize_pt2e(model, make(), [make() for _ in range(3)])
    path = tmp_path / f"{name}.pt2"
    manyseg.save(program, path, source=model, activations=activations)
    return manyseg.load(path)


def agrees_with_float(out, expected, name):
    assert out[name].shape == expected[name].shape
    assert np.isfinite(out[name]).all()
    assert (out[name].argmax(0) == expected[name].argmax(0)).mean() > 0.8


def test_saved_program_with_a_reduced_resolution_head(tmp_path):
    manyseg = pytest.importorskip("manyseg")
    torch.manual_seed(0)
    model = manyseg.create_model(
        backbone="resnet18",
        pretrained=False,
        in_channels=3,
        heads={
            "seg": {"type": "segmentation", "out_channels": 3},
            "coarse": {"type": "segmentation", "out_channels": 2, "scale": 0.25},
        },
    ).eval()
    activations = {"seg": "softmax", "coarse": "softmax"}
    program = save_and_load(model, torch.rand(1, 3, 64, 64), tmp_path, "coarse", activations)
    out = predict_array(program, scene(), output="probs", device="cpu", **KW)
    expected = predict_array(
        model, scene(), output="probs", activation="softmax", device="cpu", dtype="float32", **KW
    )
    assert out["seg"].shape == (3, 96, 112) and out["coarse"].shape == (2, 24, 28)
    agrees_with_float(out, expected, "seg")
    agrees_with_float(out, expected, "coarse")


def test_saved_program_with_two_inputs(tmp_path):
    manyseg = pytest.importorskip("manyseg")
    torch.manual_seed(0)
    model = manyseg.create_model(
        inputs={"a": {"in_channels": 3}, "b": {"in_channels": 2}},
        backbone="resnet18",
        pretrained=False,
        heads={"seg": {"type": "segmentation", "out_channels": 3}},
    ).eval()
    example = lambda: {"a": torch.rand(1, 3, 64, 64), "b": torch.rand(1, 2, 64, 64)}  # noqa: E731
    program = save_and_load(model, example, tmp_path, "two", {"seg": "softmax"})
    rng = np.random.default_rng(3)
    x = {
        "a": rng.random((3, 96, 112)).astype(np.float32),
        "b": rng.random((2, 96, 112)).astype(np.float32),
    }
    out = predict_array(program, x, output="probs", device="cpu", **KW)
    expected = predict_array(
        model, x, output="probs", activation="softmax", device="cpu", dtype="float32", **KW
    )
    assert out["seg"].shape == (3, 96, 112)
    agrees_with_float(out, expected, "seg")


def test_saved_program_compiled(saved_program, manyseg_models):
    out = predict_array(saved_program(), scene(), compile=True, output="probs", **KW)
    close_to_float(out, manyseg_models)


def test_tiling_spec_survives_quantisation(manyseg_models):
    assert manyseg_models().tiling_spec() == manyseg_models.float_model().tiling_spec()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("mode", MODES)
def test_on_the_gpu(manyseg_models, mode, dtype):
    """The crash this file was written for: int8 weights in float16 on a GPU. The outputs must
    also be the right ones, not only finite."""
    out = predict_array(manyseg_models(mode), scene(), device="cuda", dtype=dtype, **KW)
    expected = predict_array(
        manyseg_models.float_model(), scene(), device="cpu", dtype="float32", **KW
    )
    assert np.isfinite(out["seg"]).all()
    assert (out["seg"].argmax(0) == expected["seg"].argmax(0)).mean() > 0.9
    tolerance = 0.05 * np.abs(expected["seg"]).max()
    np.testing.assert_allclose(out["seg"], expected["seg"], atol=tolerance)
