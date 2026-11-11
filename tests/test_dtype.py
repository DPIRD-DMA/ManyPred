"""dtype: float32 by default, the 16-bit options, the mosaic's dtype and the warning for
infinite or NaN outputs."""

import warnings

import numpy as np
import pytest
import torch

import manypred.tiled as tiled
from manypred import predict_array, predict_batch, predict_files
from manypred.devices import inference_dtype

F16, BF16, F32 = torch.float16, torch.bfloat16, torch.float32
KW = {"device": "cpu", "progress": False}


def test_float32_is_the_default():
    model = torch.nn.Conv2d(1, 1, 1)
    weight = model.weight.detach().clone()
    x = np.ones((1, 16, 16), np.float32)
    assert predict_array(model, x, patch_size=16, overlap=0, **KW).dtype == np.float32
    assert predict_batch(model, x[None], **KW).dtype == np.float32
    assert model.weight.dtype == F32 and torch.equal(model.weight, weight)


def test_a_double_model_runs_in_float32_by_default():
    model = torch.nn.Conv2d(1, 1, 1).double()
    predict_batch(model, np.ones((1, 1, 16, 16), np.float32), **KW)
    assert model.weight.dtype == F32


def test_auto_is_no_longer_a_dtype():
    x = np.ones((1, 16, 16), np.float32)
    with pytest.raises(ValueError, match="Unknown dtype"):
        predict_array(torch.nn.Conv2d(1, 1, 1), x, dtype="auto", **KW)


@pytest.mark.filterwarnings("ignore:Inference on the CPU")
def test_explicit_dtypes_are_used_as_given():
    assert inference_dtype("bf16", torch.device("cpu")) == BF16
    assert inference_dtype(torch.float32, torch.device("cpu")) == F32
    with pytest.raises(ValueError, match="Unknown dtype"):
        inference_dtype("float8", torch.device("cpu"))
    model = torch.nn.Conv2d(1, 1, 1)
    predict_batch(model, np.ones((1, 1, 16, 16), np.float32), dtype="bfloat16", **KW)
    assert model.weight.dtype == BF16


def _rounding_warnings(model, **kwargs):
    x = np.ones((1, 1, 16, 16), np.float32)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_batch(model, x, **{**KW, **kwargs})
    return [w for w in record if "rounds its" in str(w.message)]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_16_bit_dtype_warns_about_a_model_in_training(dtype):
    found = _rounding_warnings(torch.nn.Conv2d(1, 1, 1).train(), dtype=dtype)
    assert len(found) == 1 and dtype in str(found[0].message)
    assert found[0].filename == __file__


def test_double_model_in_training_warns_by_default():
    assert len(_rounding_warnings(torch.nn.Conv2d(1, 1, 1).double().train())) == 1


@pytest.mark.parametrize(
    "make, kwargs",
    [
        (lambda: torch.nn.Conv2d(1, 1, 1).train(), {}),  # float32 stays exact
        (lambda: torch.nn.Conv2d(1, 1, 1).eval(), {"dtype": "float16"}),  # a finished model
        (lambda: torch.nn.Conv2d(1, 1, 1).half().train(), {}),  # widening loses nothing
        (lambda: torch.nn.Conv2d(1, 1, 1).half().train(), {"dtype": "float16"}),  # already cast
    ],
)
def test_no_rounding_warning_when_nothing_is_lost(make, kwargs):
    assert not _rounding_warnings(make(), **kwargs)


def test_plain_callables_run_on_float32_inputs_by_default():
    weight = torch.randn(2, 3, 1, 1)
    x = np.random.default_rng(5).random((3, 16, 16)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = predict_batch(lambda t: torch.nn.functional.conv2d(t, weight), [x], **KW)
    assert out.dtype == np.float32


def test_mosaic_is_float32_by_default_whatever_the_dtype(monkeypatch):
    made = []
    original = tiled.MosaicSink._mosaic
    monkeypatch.setattr(
        tiled.MosaicSink,
        "_mosaic",
        lambda self, name, dtype: made.append(dtype) or original(self, name, dtype),
    )
    x = np.ones((1, 64, 64), np.float32)
    kw = {"patch_size": 32, "overlap": 8, **KW}
    predict_array(torch.nn.Conv2d(1, 1, 1), x, **kw)
    with pytest.warns(UserWarning, match="on the CPU"):
        predict_array(torch.nn.Conv2d(1, 1, 1), x, dtype="float16", **kw)
    predict_array(torch.nn.Conv2d(1, 1, 1), x, mosaic_dtype="float16", **kw)
    assert made == [F32, F32, F16]


def test_large_values_stay_exact_in_the_default_mosaic():
    x = np.full((1, 96, 96), 30000.0, np.float32)
    out = predict_array(torch.nn.Identity(), x, patch_size=32, overlap=16, **KW)
    np.testing.assert_allclose(out, x, rtol=1e-6)


def test_float16_mosaic_warns_when_its_sums_overflow():
    # the weights of two patches sum to 9/8 where they overlap by 8: 67 500, past 65504
    x = np.full((1, 96, 96), 60000.0, np.float32)
    with pytest.warns(UserWarning, match="its float16 mosaic"):
        predict_array(
            torch.nn.Identity(), x, patch_size=32, overlap=8, mosaic_dtype="float16", **KW
        )


def test_opposite_overflows_in_a_float16_mosaic_warn():
    first = []

    def model(x):  # finite in float32; +inf and -inf once cast to the mosaic's float16
        sign = 1.0 if not first else -1.0
        first.append(True)
        return x[:, :1] * 0 + sign * 100000.0

    x = np.ones((1, 48, 48), np.float32)
    with pytest.warns(UserWarning, match="its float16 mosaic") as record:
        out = predict_array(
            model, x, patch_size=32, overlap=24, batch_size=1, mosaic_dtype="float16", **KW
        )
    assert np.isnan(out).any() and not _bad_warnings(record)


def test_overflowed_float16_mosaic_warns():
    sums = torch.tensor([[1.0, float("inf")]], dtype=F16)
    with pytest.warns(UserWarning, match="its float16 mosaic"):
        tiled.warn_if_overflowed(sums, "height")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiled.warn_if_overflowed(sums.float(), "height")  # float32 sums: inf is a real value
        tiled.warn_if_overflowed(torch.ones(2, dtype=F16), "height")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.cuda
@cuda
def test_auto_mosaic_device_moves_to_the_cpu_when_the_gpu_is_full(monkeypatch):
    x = np.random.default_rng(9).random((3, 300, 260)).astype(np.float32)
    kw = {"patch_size": 64, "overlap": 16, "device": "cuda", "progress": False, "dtype": "fp32"}
    model = torch.nn.Conv2d(3, 4, 1)
    on_gpu = predict_array(model, x, mosaic_device=None, **kw)
    on_cpu = predict_array(model, x, mosaic_device="cpu", **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # plenty of memory: stays on the GPU, no warning
        np.testing.assert_array_equal(predict_array(model, x, **kw), on_gpu)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1, 2**34))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    with pytest.warns(UserWarning, match="blended on the CPU") as caught:
        moved = predict_array(model, [x, x][0], **kw)
    assert len(caught) == 1 and caught[0].filename == __file__
    np.testing.assert_array_equal(moved, on_cpu)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # mosaic_device=None: the GPU whatever the memory
        np.testing.assert_array_equal(predict_array(model, x, mosaic_device=None, **kw), on_gpu)


@pytest.mark.cuda
@cuda
def test_auto_mosaic_device_with_files_warns_once(monkeypatch, tmp_path):
    from manypred import predict_files

    x = np.random.default_rng(10).random((3, 128, 96)).astype(np.float32)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1, 2**34))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    saved = []
    with pytest.warns(UserWarning, match="blended on the CPU") as caught:
        predict_files(
            torch.nn.Conv2d(3, 2, 1),
            [0, 1, 2],
            load_func=lambda i: x,
            save_func=lambda pred, meta, item, out: saved.append(pred) or item,
            patch_size=64,
            overlap=16,
            device="cuda",
            progress=False,
        )
    assert len([w for w in caught if "blended on the CPU" in str(w.message)]) == 1
    assert len(saved) == 3


@pytest.mark.cuda
@cuda
def test_float32_by_default_on_the_gpu():
    model = torch.nn.Conv2d(1, 2, 1)
    x = np.random.default_rng(11).random((1, 64, 64)).astype(np.float32)
    predict_array(model, x, patch_size=32, overlap=8, device="cuda", progress=False)
    assert model.weight.dtype == F32 and model.weight.device.type == "cuda"


class _BreaksOnLargeValues(torch.nn.Module):
    """Passes values on, with infinity where the input is over ``limit``."""

    def __init__(self, limit=500.0, bad=torch.inf):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.limit, self.bad = limit, bad

    def forward(self, x):
        return torch.where(x > self.limit, self.bad, x * self.weight)


def _bad_warnings(record):
    return [str(w.message) for w in record if "infinite or NaN outputs in" in str(w.message)]


def _late_scene():
    scene = np.ones((1, 64, 256), dtype=np.float32)
    scene[:, :, 192:] = 1000  # the last of four patches
    return scene


SCENE_KW = {"patch_size": 64, "overlap": 0, "dtype": "float32", **KW}


def test_later_bad_batch_warns_once_with_a_count():
    with pytest.warns(UserWarning, match="infinite or NaN") as record:
        out = predict_array(_BreaksOnLargeValues(), _late_scene(), **SCENE_KW)
    assert not np.isfinite(out[0, 0, 200])
    messages = _bad_warnings(record)
    assert len(messages) == 1
    assert messages[0].startswith("1 of 4 ") and "Check the model" in messages[0]


def test_16_bit_warning_suggests_float32():
    scene = np.ones((1, 64, 128), dtype=np.float32)
    scene[:, :, 64:] = 1e5  # past float16's limit once cast
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_array(_BreaksOnLargeValues(limit=1e6), scene, **{**SCENE_KW, "dtype": "float16"})
    messages = _bad_warnings(record)
    assert len(messages) == 1 and "float16" in messages[0] and "dtype='float32'" in messages[0]


def test_clean_run_does_not_warn():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_array(_BreaksOnLargeValues(limit=1e6), _late_scene(), **SCENE_KW)
    assert not _bad_warnings(record)


def test_chip_without_data_is_not_counted():
    class NanOnZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            return torch.where(x == 0, torch.nan, x * self.weight)

    chips = np.ones((3, 1, 16, 16), dtype=np.float32)
    chips[1] = 0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_batch(NanOnZero(), chips, nodata=0, dtype="float32", batch_size=1, **KW)
    assert not _bad_warnings(record)


def test_batch_mode_warns_too():
    chips = np.ones((3, 1, 16, 16), dtype=np.float32)
    chips[2] = 1000
    with pytest.warns(UserWarning, match="1 of 3 patches or chips"):
        predict_batch(_BreaksOnLargeValues(), chips, dtype="float32", batch_size=1, **KW)


def test_each_scene_of_predict_files_counts_on_its_own():
    scenes = {"clean": np.ones((1, 64, 128), dtype=np.float32), "bad": _late_scene()}
    with pytest.warns(UserWarning, match="infinite or NaN") as record:
        predict_files(
            _BreaksOnLargeValues(),
            ["bad", "clean", "bad"],
            load_func=scenes.__getitem__,
            save_func=lambda prediction, meta, item, output_dir: item,
            **SCENE_KW,
        )
    assert [m[:6] for m in _bad_warnings(record)] == ["1 of 4", "1 of 4"]


def test_replanned_scene_counts_each_patch_once():
    class Quarter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x):
            x = torch.nn.functional.avg_pool2d(x, 4, ceil_mode=True) * self.weight
            return torch.where(x > 500, torch.inf, x)

    scene = np.ones((1, 66, 264), dtype=np.float32)
    scene[:, :, 200:] = 1000
    with pytest.warns(UserWarning, match="infinite or NaN") as record:
        predict_array(Quarter(), scene, **SCENE_KW)
    messages = _bad_warnings(record)
    assert len(messages) == 1
    # 2 rows x 5 columns of windows: the first batch, abandoned to re-plan, isn't counted as well
    assert messages[0].startswith("4 of 10 ")


def test_float8_outputs_are_counted_without_crashing():
    chips = np.ones((1, 1, 2, 2), dtype=np.float32)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = predict_batch(
            lambda x: x.to(torch.float8_e4m3fn), chips, dtype="float32", output="logits", **KW
        )
    assert np.all(out == 1) and not _bad_warnings(record)
