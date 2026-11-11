"""Regression tests for fixes in batch, spec, normalize, outputs, inputs, collate and devices."""

import copy
import io
import pickle
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

import manypred
from manypred import normalize_numpy, normalize_torch, predict_array, predict_batch
from manypred.batch import BatchSink
from manypred.collate import collate
from manypred.devices import patch_for_mps, resolve_device, resolve_dtype, same_device
from manypred.inputs import describe_inputs
from manypred.normalize import build_normalizer, resolve_scene, validate_spec
from manypred.outputs import TilePredictions
from manypred.spec import get_tiling_spec, validate_tiling_spec

KW = {"progress": False, "device": "cpu"}


class Recorder(torch.nn.Module):
    """Returns its input, so tests see what the model was given."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return x


# --------------------------------------------------------------------- single-input specs


def test_a_spec_naming_the_single_input_accepts_a_bare_array():
    spec = {"inputs": {"s2": {"kind": "raster", "scale": 1.0}}, "outputs": {"seg": {}}}
    image = np.random.rand(3, 64, 64).astype(np.float32)
    out = predict_array(Recorder(), image, spec=spec, patch_size=32, overlap=8, **KW)
    np.testing.assert_allclose(out, image, rtol=1e-6)


@pytest.mark.parametrize("entry", ["predict_batch", "predict_array"])
def test_a_per_input_stored_normalize_applies_to_a_bare_array(entry):
    spec = {
        "inputs": {"s2": {"kind": "raster"}},
        "outputs": {"seg": {}},
        "normalize": {"s2": {"type": "scale", "value": 100}},
    }
    if entry == "predict_batch":
        chips = np.full((2, 3, 8, 8), 50, dtype=np.float32)
        out = predict_batch(Recorder(), chips, spec=spec, **KW)
    else:
        chips = np.full((3, 32, 32), 50, dtype=np.float32)
        out = predict_array(Recorder(), chips, spec=spec, patch_size=32, overlap=0, **KW)
    np.testing.assert_allclose(out, 0.5)


def test_a_square_odd_scene_takes_a_half_resolution_input():
    for height, half in ((103, 52), (103, 51), (77, 39)):
        _, infos = describe_inputs(
            {"a": np.zeros((1, height, height)), "b": np.zeros((1, half, half))}
        )
        assert infos["b"].scale == pytest.approx(0.5)


def test_other_ratios_are_still_refused():
    with pytest.raises(ValueError, match="whole numbers"):
        describe_inputs({"a": np.zeros((1, 99, 99)), "b": np.zeros((1, 66, 66))})


# --------------------------------------------------------------------- normalisation


def test_false_means_no_steps_outside_the_pipeline():
    x = torch.rand(2, 4, 4)
    torch.testing.assert_close(normalize_torch(x, False), x)
    np.testing.assert_allclose(normalize_numpy(x.numpy(), False), x.numpy())


def test_scene_percentile_of_a_constant_large_band_builds():
    resolved = resolve_scene(
        {"type": "percentile", "scope": "scene"}, torch.full((1, 8, 8), 1e8), None
    )
    assert resolved[0]["max"][0] > resolved[0]["min"][0]
    x = torch.full((1, 1, 8, 8), 1e8)
    out = build_normalizer(resolved)(x, torch.ones_like(x, dtype=torch.bool))
    assert torch.isfinite(out).all()


def test_large_integer_nodata_is_compared_exactly():
    a = np.array([[[16777216, 16777217]]], dtype=np.int32)
    np.testing.assert_array_equal(normalize_numpy(a, None, nodata=16777217), [[[16777216, 0]]])
    chips = np.array([[[[16777216, 16777217]]]], dtype=np.int32)
    out = predict_batch(Recorder(), chips, output="logits", nodata=16777217, **KW)
    assert out[0, 0, 0, 0] == 16777216
    assert np.isnan(out[0, 0, 0, 1])


def test_float64_nodata_is_compared_before_float32_rounding():
    chips = np.array([[[[1.0 + 1e-12, 1.0]]]], dtype=np.float64)
    out = predict_batch(Recorder(), chips, output="logits", nodata=1.0, **KW)
    assert not np.isnan(out[0, 0, 0, 0])
    assert np.isnan(out[0, 0, 0, 1])


def test_uint16_nodata_still_matches():
    chips = np.array([[[[65535, 7]]]], dtype=np.uint16)
    out = predict_batch(Recorder(), chips, output="logits", nodata=65535, **KW)
    assert np.isnan(out[0, 0, 0, 0]) and out[0, 0, 0, 1] == 7


def test_float32_overflow_of_integer_input_becomes_zero():
    spec = {"type": "scale", "value": 1e-40}
    ones = np.ones((1, 2, 2), dtype=np.uint8)
    np.testing.assert_array_equal(normalize_numpy(ones, spec), 0)
    model = Recorder()
    predict_batch(model, ones[None], normalize=spec, output="logits", **KW)
    assert torch.isfinite(model.seen[0]).all()


def test_clip_bounds_of_different_lengths_are_explained():
    with pytest.raises(ValueError, match="same number of values"):
        validate_spec({"type": "clip", "min": [0, 0, 0], "max": [1, 1]})


def test_percentile_bounds_must_be_numbers():
    with pytest.raises(ValueError, match="percentile"):
        validate_spec({"type": "percentile", "low": "2"})


def test_validate_spec_is_exported():
    assert manypred.validate_spec is validate_spec
    assert "TilingSpec" in manypred.__all__


# --------------------------------------------------------------------- outputs


def test_a_named_setting_for_an_unnamed_single_output_is_refused():
    chips = np.random.rand(2, 3, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="single unnamed"):
        predict_batch(torch.nn.Conv2d(3, 4, 1), chips, output={"typo": "argmax"}, **KW)


def test_to_map_is_nan_at_nodata():
    valid_map = np.ones((4, 4), dtype=bool)
    valid_map[0, :] = False
    tp = TilePredictions(
        values=np.ones((1, 2), dtype=np.float32),
        windows=np.array([[0, 4, 0, 4]]),
        valid=np.array([0.75]),
        shape=(4, 4),
        weights=np.ones((4, 4), dtype=np.float32),
        valid_map=valid_map,
    )
    full = tp.to_map()
    assert np.isnan(full[:, 0]).all() and (full[:, 1:] == 1).all()
    assert np.isnan(tp.to_map(stride=2)[:, 0]).all()
    assert (tp.to_map(rows=(1, 3)) == 1).all()
    np.testing.assert_allclose(tp.reduce("mean"), 1)


def test_postprocess_keeps_no_outputs(monkeypatch):
    seen = []
    original = BatchSink.__call__

    def spy(self, prediction):
        original(self, prediction)
        seen.append((dict(self.arrays), dict(self.parts)))

    monkeypatch.setattr(BatchSink, "__call__", spy)
    chips = np.random.rand(5, 3, 4, 4).astype(np.float32)
    for data in (chips, iter(list(chips))):
        seen.clear()
        out = predict_batch(Recorder(), data, batch_size=2, postprocess=np.copy, **KW)
        np.testing.assert_allclose(np.stack(out), chips, rtol=1e-6)
        assert seen and all(arrays == {} and parts == {} for arrays, parts in seen)


@pytest.mark.parametrize("group_by_shape", [False, True])
def test_a_slow_postprocess_holds_back_prediction(group_by_shape):
    """Chips waiting for postprocess stay within 2 x batch_size, however slow it is."""
    release = threading.Event()
    calls = []

    def slow(chip):
        release.wait()
        return chip.shape

    class Counting(torch.nn.Module):
        def forward(self, x):
            calls.append(len(x))
            return x

    chips = list(np.random.rand(100, 1, 4, 4).astype(np.float32))
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(
            out=predict_batch(
                Counting(),
                chips,
                batch_size=4,
                postprocess=slow,
                group_by_shape=group_by_shape,
                **KW,
            )
        )
    )
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and sum(calls) < 12:
        time.sleep(0.01)
    time.sleep(0.3)  # give an unbounded queue time to run ahead
    predicted = sum(calls)
    release.set()
    worker.join(timeout=30)
    # 8 waiting (one running) plus a batch being finished, and one prefetched
    assert predicted <= 2 * 4 + 1 + 4 + 4, predicted
    assert len(result["out"]) == 100


def test_postprocess_can_take_the_device_lock(monkeypatch):
    """The wait for postprocess happens outside the device lock, so a postprocess that takes
    it (as an MPS prediction does) can't deadlock. Every run takes the real MPS lock here."""
    import manypred.runner as runner
    from manypred.devices import _MPS_LOCK

    monkeypatch.setattr(runner, "device_lock", lambda *devices: _MPS_LOCK)
    acquired = []

    def locking(chip):
        got = _MPS_LOCK.acquire(timeout=2)
        acquired.append(got)
        if got:
            _MPS_LOCK.release()
        return chip.shape

    result = {}
    chips = list(np.random.rand(6, 1, 4, 4).astype(np.float32))
    worker = threading.Thread(
        target=lambda: result.update(
            out=predict_batch(Recorder(), chips, batch_size=1, postprocess=locking, **KW)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive()
    assert acquired == [True] * 6
    assert len(result["out"]) == 6


def test_batch_mode_files_postprocess_can_take_the_device_lock(monkeypatch, tmp_path):
    """predict_files(mode="batch") waits for a full save queue outside the device lock too."""
    import manypred.runner as runner
    from manypred import predict_files
    from manypred.devices import _MPS_LOCK

    monkeypatch.setattr(runner, "device_lock", lambda *devices: _MPS_LOCK)
    acquired = []

    def locking(prediction):
        got = _MPS_LOCK.acquire(timeout=2)
        acquired.append(got)
        if got:
            _MPS_LOCK.release()
        return prediction

    chips = list(np.random.rand(6, 1, 4, 4).astype(np.float32))
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(
            out=predict_files(
                torch.nn.Identity(),
                list(range(6)),
                load_func=lambda i: (chips[i], None),
                save_func=lambda prediction, meta, item, output_dir: item,
                output_dir=tmp_path,
                mode="batch",
                batch_size=3,  # several saves queued while one batch holds the lock
                max_saves_in_flight=1,
                postprocess=locking,
                **KW,
            )
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive()
    assert acquired == [True] * 6
    assert len(result["out"]) == 6


_PREDICT_IN_POSTPROCESS = """
import numpy as np, torch
from manypred import predict_batch

inner = torch.nn.Conv2d(1, 1, 1)

def predict_again(chip):
    return predict_batch(inner, chip[None], device="mps", progress=False).shape

chips = list(np.random.rand(6, 1, 4, 4).astype(np.float32))
out = predict_batch(torch.nn.Identity(), chips, batch_size=1, postprocess=predict_again,
                    device="mps", progress=False)
assert out == [(1, 1, 4, 4)] * 6, out
"""


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_postprocess_can_predict_on_mps():
    """A real MPS prediction inside postprocess, in a subprocess so a deadlock fails the test
    (on the timeout) instead of hanging the test run."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PREDICT_IN_POSTPROCESS],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("deadlocked: postprocess waited on the device lock")
    assert done.returncode == 0, done.stderr


def test_a_single_chip_dict_is_explained():
    chip = {"a": np.zeros((3, 8, 8), np.float32), "meta": np.zeros(4, np.float32)}
    with pytest.raises(ValueError, match="single chip"):
        predict_batch(Recorder(), chip, **KW)


# --------------------------------------------------------------------- tiling spec


@pytest.mark.parametrize(
    "spec",
    [
        {"inputs": {"a": {"kind": "raster", "scael": 1.0}}},
        {"outputs": {"seg": {"activaton": "softmax"}}},
        {"pad_multiple": True},
        {"max_size": True},
        {"max_size": (True, 64)},
    ],
)
def test_bad_tiling_specs_are_refused(spec):
    with pytest.raises(ValueError):
        validate_tiling_spec(spec)


def test_an_ensemble_with_some_specs_says_so():
    class Declares(torch.nn.Identity):
        def tiling_spec(self):
            return {"outputs": {"seg": {}}}

    with pytest.raises(ValueError, match="1 of the 2 models"):
        get_tiling_spec([Declares(), torch.nn.Identity()], None)


# --------------------------------------------------------------------- devices


def test_float64_by_name():
    for name in ("float64", "fp64", "double", "torch.float64"):
        assert resolve_dtype(name) is torch.float64


def test_same_device_ignores_a_missing_index():
    assert same_device(torch.device("mps"), torch.device("mps:0"))
    assert same_device(torch.device("cpu"), torch.device("cpu"))
    assert not same_device(torch.device("cpu"), torch.device("mps"))


def test_resolve_device_pins_cuda_to_the_callers_gpu(monkeypatch):
    # the current GPU is per thread: worker threads start on GPU 0, so "cuda" has to be
    # resolved to the caller's GPU before they start
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert resolve_device("cuda") == torch.device("cuda", 1)
    assert resolve_device(torch.device("cuda")) == torch.device("cuda", 1)
    assert resolve_device(None) == torch.device("cuda", 1)
    assert resolve_device("cuda:0") == torch.device("cuda", 0)
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_leaves_cuda_alone_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == torch.device("cuda")


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two GPUs"
)
def test_cuda_means_the_current_gpu_on_every_thread():
    previous = torch.cuda.current_device()
    torch.cuda.set_device(1)
    try:
        model = torch.nn.Conv2d(2, 1, 1).to("cuda:1").eval()
        chips = [torch.rand(2, 16, 16, device="cuda:1") for _ in range(6)]
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        before = torch.cuda.memory_allocated(0)
        out = predict_batch(model, chips, device="cuda", batch_size=2)
        torch.cuda.synchronize(0)
        assert torch.cuda.max_memory_allocated(0) == before  # nothing staged on GPU 0
        expected = model(torch.stack(chips)).detach().cpu().numpy()
        np.testing.assert_allclose(np.stack(out), expected, rtol=1e-5, atol=1e-6)
    finally:
        torch.cuda.set_device(previous)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_a_batch_on_the_device_is_used_as_it_is():
    batch = torch.rand(2, 3, 4, 4, device="mps:0")
    out, _ = collate(batch, torch.device("mps"))
    assert out is batch


def test_mps_patched_models_pickle_as_plain_conv2d():
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
    patch_for_mps([model])
    assert type(model[0]) is not torch.nn.Conv2d
    for restored in (
        pickle.loads(pickle.dumps(model)),
        copy.deepcopy(model),
        _torch_round_trip(model),
    ):
        assert type(restored[0]) is torch.nn.Conv2d
        torch.testing.assert_close(restored[0].weight, model[0].weight)


def _torch_round_trip(model):
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


# --------------------------------------------------------------------- scene statistics


def _scene():
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 10000, size=(3, 37, 29)).astype(np.uint16)
    scene[:, :5] = 65535
    return scene


@pytest.mark.parametrize(
    "spec",
    [
        [{"type": "scale", "value": 100}, {"type": "zscore", "scope": "scene"}],
        {"type": "percentile", "low": 5, "high": 95, "scope": "scene"},
    ],
)
def test_scene_statistics_stream_in_row_chunks(monkeypatch, spec):
    import manypred.normalize as normalize

    scene = _scene()
    whole = resolve_scene(spec, torch.from_numpy(scene.astype(np.float32)), 65535)
    sizes = []
    original = normalize._scene_chunks

    def spy(*args):
        for x, valid in original(*args):
            sizes.append(x.numel())
            yield x, valid

    monkeypatch.setattr(normalize, "SCENE_CHUNK", 3 * 29 * 4)
    monkeypatch.setattr(normalize, "_scene_chunks", spy)
    chunked = resolve_scene(spec, scene, 65535)  # the uint16 array itself, no float32 copy
    assert max(sizes) <= 3 * 29 * 4 and len(sizes) > 1
    last = chunked[-1]
    for key in ("mean", "std", "min", "max"):
        if key in last:
            np.testing.assert_allclose(last[key], whole[-1][key], rtol=1e-5)
    valid = scene[0, 5:].astype(np.float64) / 100
    if "mean" in last:
        np.testing.assert_allclose(last["mean"][0], valid.mean(), rtol=1e-5)
        np.testing.assert_allclose(last["std"][0], valid.std(), rtol=1e-5)
    else:
        np.testing.assert_allclose(
            [last["min"][0], last["max"][0]], np.percentile(scene[0, 5:], [5, 95]), rtol=1e-6
        )


def test_scene_scope_of_an_integer_array_matches_float32():
    scene = _scene()
    spec = {"type": "zscore", "scope": "scene"}
    kw = {"patch_size": 32, "overlap": 8, "nodata": 65535, **KW}
    a = predict_array(Recorder(), scene, normalize=spec, **kw)
    b = predict_array(Recorder(), scene.astype(np.float32), normalize=spec, **kw)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)


# --------------------------------------------------------------------- reduced outputs


class _Reduce4(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x, 4, ceil_mode=True)


def test_an_ambiguous_output_factor_is_an_error():
    chips = np.ones((1, 1, 5, 5), np.float32)
    with pytest.raises(ValueError, match=r"any of \[3, 4\]"):
        predict_batch(_Reduce4(), chips, output="logits", **KW)


def test_the_declared_output_factor_masks_nodata():
    chips = np.full((1, 1, 5, 5), np.nan, np.float32)
    chips[..., 3] = 1  # valid only in column 3: block 0 for f = 4, block 1 for f = 3
    spec = {"outputs": {"seg": {"scale": 0.25}}}
    out = predict_batch(_Reduce4(), chips, output="logits", spec=spec, **KW)
    assert not np.isnan(out[0, 0, :, 0]).any()
    assert np.isnan(out[0, 0, :, 1]).all()


def test_a_one_pixel_output_needs_no_factor():
    class Pool(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

    out = predict_batch(Pool(), np.ones((2, 1, 5, 5), np.float32), output="logits", **KW)
    assert out.shape == (2, 1, 1, 1)


def test_int64_nodata_past_2_53_is_exact():
    values = torch.tensor([[[2**53, 2**53 + 1]]], dtype=torch.int64)
    from manypred.normalize import band_valid_mask
    from manypred.windows import nodata_pixels

    assert band_valid_mask(values, 2**53 + 1).tolist() == [[[True, False]]]
    assert band_valid_mask(values, np.int64(2**53 + 1)).tolist() == [[[True, False]]]
    assert nodata_pixels(values, 2**53 + 1).tolist() == [[False, True]]
    assert nodata_pixels(values.numpy(), 2**53 + 1).tolist() == [[False, True]]
    out = normalize_numpy(values.numpy(), None, nodata=2**53 + 1)
    assert out[0, 0, 0] != 0 and out[0, 0, 1] == 0


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "spec", [{"type": "zscore", "scope": "scene"}, {"type": "percentile", "scope": "scene"}]
)
def test_scene_scope_of_an_mps_tensor(spec):
    scene = torch.from_numpy(_scene().astype(np.float32))
    kw = {"patch_size": 32, "overlap": 8, "nodata": 65535, "progress": False}
    on_mps = predict_array(Recorder(), scene.to("mps"), normalize=spec, device="mps", **kw)
    on_cpu = predict_array(Recorder(), scene, normalize=spec, device="cpu", **kw)
    np.testing.assert_allclose(on_mps, on_cpu, rtol=1e-4, atol=1e-5, equal_nan=True)


def test_reduce_mean_counts_each_channel_on_its_own():
    """A NaN in channel 0 dropped the pixel from every channel's mean."""
    tp = TilePredictions(
        values=np.array([[np.nan, 2], [3, 4]], dtype=np.float32),
        windows=np.array([[0, 1, 0, 1], [0, 1, 1, 2]]),
        valid=np.array([1.0, 1.0]),
        shape=(1, 2),
        weights=np.ones((1, 1), dtype=np.float32),
    )
    np.testing.assert_allclose(tp.reduce("mean"), [3, 3])
    nan_only = TilePredictions(
        values=np.array([[np.nan, 1]], dtype=np.float32),
        windows=np.array([[0, 1, 0, 1]]),
        valid=np.array([1.0]),
        shape=(1, 1),
        weights=np.ones((1, 1), dtype=np.float32),
    )
    out = nan_only.reduce("mean")
    assert np.isnan(out[0]) and out[1] == 1 and out.dtype == np.float32


# --------------------------------------------------------------------- second sweep


def _chip_mean(x):
    return x.float().mean(dim=(2, 3))[:, :1]


@pytest.mark.parametrize("ids", [np.array([0, 1, 2]), torch.tensor([0, 1, 2]), [0, 1, 2]])
def test_grouped_items_in_an_array_keep_their_order(ids):
    sizes = {0: (4, 1.0), 1: (5, 2.0), 2: (4, 3.0)}

    def load(i):
        side, value = sizes[int(i)]
        return np.full((1, side, side), value, dtype=np.float32)

    out = predict_batch(
        _chip_mean, ids, load_func=load, group_by_shape=True, batch_size=2, output="logits", **KW
    )
    assert [float(np.asarray(x).ravel()[0]) for x in out] == [1.0, 2.0, 3.0]


def test_mixed_integer_vectors_stay_integers():
    image = np.zeros((1, 4, 4), dtype=np.float32)
    chips = [
        {"x": image, "v": np.array([16777217], dtype=np.int32)},
        {"x": image, "v": np.array([16777217], dtype=np.int64)},
    ]

    def model(inputs):
        return (inputs["v"].long() == 16777217).float()

    for batch_size in (1, 2):
        out = predict_batch(model, chips, batch_size=batch_size, output="logits", **KW)
        np.testing.assert_array_equal(out.ravel(), [1, 1])


def test_mixed_raster_dtypes_share_one_that_holds_them():
    batch, _ = collate(
        [np.full((1, 2, 2), 70000, np.int32), np.zeros((1, 2, 2), np.uint8)], torch.device("cpu")
    )
    assert batch.dtype == torch.int32 and int(batch.max()) == 70000


def test_mixed_uint8_and_uint16_rasters_keep_their_values():
    batch, is_uint16 = collate(
        [np.full((1, 2, 3), 3, np.uint8), np.full((1, 2, 3), 500, np.uint16)], torch.device("cpu")
    )
    from manypred.collate import to_float32

    values = to_float32(batch, is_uint16)
    assert values[0].unique().tolist() == [3.0] and values[1].unique().tolist() == [500.0]


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        ((np.int32, np.float32), torch.float64),
        ((np.int32, np.float64), torch.float64),
        ((np.uint8, np.float16), torch.float16),
    ],
)
def test_mixed_int_and_float_vectors(dtypes, expected):
    vectors = [np.array([1], dtypes[0]), np.array([0.5], dtypes[1])]
    batch, _ = collate(vectors, torch.device("cpu"))
    assert batch.dtype == expected and batch.ravel().tolist() == [1.0, 0.5]


def test_int64_and_uint64_vectors_stay_integers():
    batch, _ = collate([np.array([2**40], np.int64), np.array([7], np.uint64)], torch.device("cpu"))
    assert batch.dtype == torch.int64 and batch.ravel().tolist() == [2**40, 7]


def test_uint64_nodata_is_exact():
    scene = np.full((1, 4, 4), 16777216, dtype=np.uint64)
    scene[0, 0, 0] = 16777217

    def ones(x):
        return torch.ones(x.shape[0], 1, *x.shape[2:])

    tiled = predict_array(
        ones, scene, patch_size=4, overlap=0, nodata=16777217, output="logits", **KW
    )
    chips = predict_batch(ones, scene[None], nodata=16777217, output="logits", **KW)
    assert np.isnan(tiled).sum() == 1
    assert np.isnan(chips).sum() == 1


def test_uint64_past_int64_travels_as_float():
    big = np.array([[[2**64 - 1, 5]]], dtype=np.uint64)
    batch, _ = collate([big], torch.device("cpu"))
    assert batch.dtype == torch.float64 and float(batch[0, 0, 0, 1]) == 5


def test_reduce_max_skips_nan_per_channel():
    tiles = TilePredictions(
        values=np.array([[0.2, np.nan], [0.4, 0.9]], dtype=np.float32),
        windows=np.array([[0, 4, 0, 4], [0, 4, 4, 8]]),
        valid=np.array([1.0, 0.25], dtype=np.float32),
        shape=(4, 8),
        weights=np.ones((4, 4), dtype=np.float32),
    )
    np.testing.assert_allclose(tiles.reduce("max"), [0.4, 0.9])
    np.testing.assert_allclose(tiles.reduce("max", min_valid=0.5), [0.2, np.nan])


class _TwoHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1)

    def forward(self, x):
        y = self.conv(x)
        return {"seg": y, "coarse": y[..., ::4, ::4], "score": y.mean(dim=(2, 3))}


@pytest.mark.parametrize("mode", ["tiled", "batch"])
def test_save_func_factors_are_the_same_in_both_modes(mode):
    got = {}

    def save(prediction, meta, item, output_dir, factors=None):
        got[item] = factors
        return item

    images = {"a": np.random.default_rng(0).random((2, 32, 32), dtype=np.float32)}
    manypred.predict_files(
        _TwoHeads(),
        ["a"],
        load_func=images.get,
        save_func=save,
        mode=mode,
        patch_size=16,
        overlap=4,
        output="logits",
        **KW,
    )
    assert got["a"] == {"seg": 1, "coarse": 4, "score": None}


@pytest.mark.parametrize("lazy", [False, True])
def test_block_fields_may_name_a_bare_heads_spec_name(tmp_path, lazy):
    pytest.importorskip("rasterio")
    from manypred.lazy import ArrayReader

    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

        def tiling_spec(self):
            return {"outputs": {"seg": {"activation": "softmax"}}}

    def fn(fields, block, scene):
        return sorted(fields)

    image = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32)
    post = manypred.BlockPostprocess(fn, block=16, halo=2, fields=["seg"])
    manypred.predict_files(
        Bare(),
        [tmp_path / "a.tif"],
        load_func=lambda item: ArrayReader(image) if lazy else image,
        output_dir=tmp_path / "out",
        patch_size=16,
        overlap=4,
        block_postprocess=post,
        **KW,
    )
    assert post.results[0] == [["seg"]] * 4  # under the name fields= gives


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_float64_vectors_on_mps():
    image = np.zeros((1, 4, 4), np.float32)
    chips = [{"x": image, "v": np.array([0.5])}, {"x": image, "v": np.array([2], np.int32)}]
    for batch_size in (1, 2):
        out = predict_batch(
            lambda d: d["v"][:, :1].float() + d["x"].mean(dim=(1, 2, 3))[:, None],
            chips,
            batch_size=batch_size,
            output="logits",
            progress=False,
            device="mps",
        )
        np.testing.assert_allclose(out.ravel(), [0.5, 2.0])


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_tile_percentile_on_mps():
    chips = np.random.default_rng(0).random((2, 1, 8, 8), dtype=np.float32)
    out = predict_batch(
        lambda x: x,
        chips,
        normalize={"type": "percentile"},
        output="logits",
        progress=False,
        device="mps",
    )
    expected = normalize_numpy(chips, {"type": "percentile"})
    np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, 0, 5), (2, 5, 0), (0, 5, 5)])
def test_empty_image_is_refused_in_tiled_mode(shape):
    with pytest.raises(ValueError, match="empty"):
        predict_array(
            torch.nn.Conv2d(2, 3, 1),
            np.ones(shape, np.float32),
            patch_size=8,
            overlap=2,
            **KW,
        )


def test_mixed_int32_and_float32_rasters_compare_nodata_exactly():
    # int32 with float32 must not round 16777217 to the nodata value 16777216 when batched
    chips = [np.full((1, 2, 2), 16777217, np.int32), np.ones((1, 2, 2), np.float32)]

    def ones(x):
        return torch.ones(x.shape[0], 1, *x.shape[2:])

    for batch_size in (1, 2):
        out = predict_batch(
            ones, chips, batch_size=batch_size, nodata=16777216, output="logits", **KW
        )
        np.testing.assert_array_equal(out, 1)


def test_a_wrong_declared_output_scale_is_refused_in_both_modes():
    image = np.random.default_rng(0).random((1, 8, 8), dtype=np.float32)
    spec = {"outputs": {"seg": {"scale": 0.25}}}

    def half(x):
        return x[:, :, ::2, ::2]

    with pytest.raises(ValueError, match="declares it at 1/4"):
        predict_array(half, image, patch_size=8, overlap=0, spec=spec, **KW)
    with pytest.raises(ValueError, match="Check the output's scale"):
        predict_batch(half, image[None], spec=spec, **KW)
    good = {"outputs": {"seg": {"scale": 0.5}}}
    assert predict_array(half, image, patch_size=8, overlap=0, spec=good, **KW).shape == (1, 4, 4)


def test_fields_of_a_reduced_head_are_masked_on_its_grid():
    from manypred.runner import set_up_run
    from manypred.tiled import Tiler

    setup = set_up_run(
        lambda x: x[:, :, ::2, ::2],
        device="cpu",
        dtype="float32",
        batch_size=1,
        default_batch_size=1,
        spec=None,
        normalize=None,
        activation=None,
        output="argmax",
        nodata=None,
        compile=False,
    )
    tiler = Tiler(setup.runner, 4, 0, 1, None, True, False)
    image = np.random.default_rng(0).random((2, 4, 4), dtype=np.float32)
    result, _, fields = tiler.predict_fields(image, lambda factors, alias: [""])
    assert fields[""].shape == (2, 2, 2) and result[""].shape == (1, 2, 2)


@pytest.mark.parametrize(
    ("chips", "nodata", "expected"),
    [
        # float32 0.1 is nodata 0.1 alone, so batched with int32 (as float64) too
        ([np.full((1, 2, 2), 0.1, np.float32), np.ones((1, 2, 2), np.int32)], 0.1, [1, 0]),
        ([np.full((1, 2, 2), 0.1, np.float16), np.full((1, 2, 2), 0.1, np.float64)], 0.1, [1, 1]),
        # an integer chip can't hold 5.5, so it never matches it
        ([np.full((1, 2, 2), 5, np.int16), np.full((1, 2, 2), 5.5, np.float32)], 5.5, [0, 1]),
    ],
)
def test_nodata_of_mixed_dtype_chips_is_matched_as_each_dtype_holds_it(chips, nodata, expected):
    def ones(x):
        return torch.ones(x.shape[0], 1, *x.shape[2:])

    for batch_size in (1, 2):
        out = predict_batch(
            ones, chips, batch_size=batch_size, nodata=nodata, output="logits", **KW
        )
        assert [int(np.isnan(np.asarray(o)).all()) for o in out] == expected


@pytest.mark.parametrize(
    "model",
    [lambda x: x, lambda x: x[..., :1, :1], lambda x: x[..., ::2, ::2]],
    ids=["full", "one-pixel", "half"],
)
def test_a_wrong_declared_scale_is_refused_by_every_path(model):
    image = np.random.default_rng(0).random((1, 8, 8), dtype=np.float32)
    spec = {"outputs": {"seg": {"scale": 0.25}}}
    with pytest.raises(ValueError):
        predict_array(model, image, patch_size=8, overlap=0, spec=spec, **KW)
    with pytest.raises(ValueError):
        predict_batch(model, image[None], spec=spec, **KW)
    with pytest.raises(ValueError):
        predict_batch(model, image[None], spec=spec, mask_nodata=False, output="logits", **KW)


def test_int64_max_nodata_in_a_mixed_batch():
    chips = [np.full((1, 2, 2), 2**63 - 1, np.int64), np.ones((1, 2, 2), np.float32)]

    def ones(x):
        return torch.ones(x.shape[0], 1, *x.shape[2:])

    for batch_size in (1, 2):
        out = predict_batch(
            ones, chips, batch_size=batch_size, nodata=2**63 - 1, output="logits", **KW
        )
        assert [bool(np.isnan(np.asarray(o)).all()) for o in out] == [True, False]


def test_an_explicit_full_scale_is_checked():
    image = np.random.default_rng(0).random((1, 8, 8), dtype=np.float32)
    spec = {"outputs": {"seg": {"scale": 1.0}}}

    def half(x):
        return x[..., ::2, ::2]

    with pytest.raises(ValueError):
        predict_array(half, image, patch_size=8, overlap=0, spec=spec, **KW)
    with pytest.raises(ValueError):
        predict_batch(half, image[None], spec=spec, **KW)
    assert predict_batch(lambda x: x, image[None], spec=spec, **KW).shape == (1, 1, 8, 8)


def test_programs_on_one_device_named_two_ways_agree():
    from manypred.runner import program_device

    class Program:
        def __init__(self, device):
            self.device = device

        def __call__(self, x):
            return x

    assert program_device([Program("cpu"), Program("cpu:0")], None) == torch.device("cpu")
    with pytest.raises(ValueError, match="different device"):
        program_device([Program("cpu"), Program("meta")], None)


@pytest.mark.parametrize("mode", ["batch", "tiled"])
def test_declared_vectors_stay_float32_under_16_bit_dtypes(mode):
    seen = {}

    class Model(torch.nn.Module):
        def forward(self, inputs):
            seen["meta"] = inputs["meta"].clone()
            return inputs["img"][:, :1].float()

        def tiling_spec(self):
            return {"inputs": {"img": {}, "meta": {"kind": "vector"}}}

    class Plain(Model):
        tiling_spec = None

    meta = np.array([257.0, 52.13], np.float32)
    image = np.ones((2, 8, 8), np.float32)
    for model, exact in ((Model(), True), (Plain(), False)):
        if mode == "batch":
            chips = {"img": image[None], "meta": meta[None]}
            predict_batch(model, chips, dtype="bfloat16", output="logits", **KW)
        else:
            scene = {"img": image, "meta": meta}
            predict_array(
                model, scene, dtype="bfloat16", patch_size=8, overlap=0, output="logits", **KW
            )
        if exact:  # the spec declares it: exact, in float32
            assert seen["meta"].dtype == torch.float32
            np.testing.assert_array_equal(seen["meta"].numpy().ravel(), meta)
        else:  # no spec: cast like any input, so a plain bfloat16 layer can take it
            assert seen["meta"].dtype == torch.bfloat16
