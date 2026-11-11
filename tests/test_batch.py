"""predict_batch behaviour, ported from OmniCloudMask's predict_from_batch tests where generic."""

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from helpers import (
    ALL_DEVICES,
    DEVICES,
    IMAGE_SIZE,
    FailOnCall,
    SeededConv,
    random_images,
    run_with_timeout,
    striped_images,
    wait_for_new_threads,
)

from manypred import predict_batch
from manypred.collate import collate, to_float32
from manypred.runner import RawBatch, prefetch

KW = {"normalize": {"type": "zscore", "scope": "tile"}, "progress": False}


def predict(data, device="cpu", model=None, **kwargs):
    return predict_batch(model or SeededConv(), data, device=device, **{**KW, **kwargs})


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    return random_images(5)


@pytest.fixture(scope="module")
def reference(images) -> np.ndarray:
    return predict(images, batch_size=2, output="argmax")


def test_output_shapes_and_dtypes(images, reference):
    assert reference.shape == (5, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert reference.dtype == np.uint8
    logits = predict(images, batch_size=2)
    assert logits.shape == (5, 4, IMAGE_SIZE, IMAGE_SIZE) and logits.dtype == np.float32
    probs = predict(images, batch_size=2, activation="softmax", output="probs")
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_batch_size_invariant(images, reference, batch_size):
    np.testing.assert_array_equal(
        predict(images, batch_size=batch_size, output="argmax"), reference
    )


@pytest.mark.parametrize(
    "make_data",
    [
        pytest.param(list, id="list_of_arrays"),
        pytest.param(lambda x: (image for image in x), id="generator"),
        pytest.param(torch.from_numpy, id="tensor"),
        pytest.param(lambda x: [torch.from_numpy(i) for i in x], id="list_of_tensors"),
        pytest.param(
            lambda x: [torch.from_numpy(i) if n % 2 else i for n, i in enumerate(x)],
            id="mixed_list",
        ),
    ],
)
def test_input_types(images, reference, make_data):
    np.testing.assert_array_equal(
        predict(make_data(images), batch_size=2, output="argmax"), reference
    )


def test_load_func(images, reference):
    np.testing.assert_array_equal(
        predict(range(5), load_func=lambda i: images[i], batch_size=2, output="argmax"),
        reference,
    )
    np.testing.assert_array_equal(
        predict(range(5), load_func=lambda i: (images[i], {"i": i}), batch_size=2, output="argmax"),
        reference,
    )


def test_nodata_mask(images):
    data = images[:3].copy()
    data[0, :, :10, :] = 0  # partly nodata
    data[1] = 0  # entirely nodata
    classes = predict(data, batch_size=3, output="argmax", nodata=0)
    assert np.all(classes[0, :, :10] == 0)
    assert np.all(classes[1] == 0)
    logits = predict(data, batch_size=3, nodata=0)
    assert np.isnan(logits[0, :, :10]).all()
    # nothing is predicted for an all-nodata chip, as for an all-nodata patch
    assert np.isnan(logits[1]).all()
    assert not np.isnan(logits[0, :, 10:]).any()
    assert not np.isnan(logits[2]).any()


def test_without_nodata_mask(images):
    data = images[:3].copy()
    data[0, :, :10, :] = 0
    data[1] = 0
    kwargs = {"batch_size": 3, "activation": "softmax", "output": "probs", "nodata": 0}
    unmasked = predict(data, mask_nodata=False, **kwargs)
    masked = predict(data, mask_nodata=True, **kwargs)
    assert unmasked[0, :, :10].min() > 0
    assert np.isnan(masked[0, :, :10]).all()
    np.testing.assert_array_equal(unmasked[0, :, 10:], masked[0, :, 10:])
    np.testing.assert_array_equal(unmasked[2], masked[2])
    assert np.isnan(unmasked[1]).all()


def test_nan_nodata(images):
    data = images[:2].copy()
    data[0, :, :10, :] = np.nan
    result = predict(data, batch_size=2, nodata=np.nan, output="argmax")
    assert np.all(result[0, :, :10] == 0)
    logits = predict(data, batch_size=2, nodata=np.nan)
    assert np.isnan(logits[0, :, :10]).all()
    assert not np.isnan(logits[0, :, 10:]).any()
    assert not np.isnan(logits[1]).any()


def test_nodata_none_uses_every_pixel(images):
    data = images[:1].copy()
    data[0, :, :10, :] = 0
    logits = predict(data, nodata=None)
    assert not np.all(logits[0, :, :10] == 0)


def test_shape_mismatch(images):
    data = [images[0], images[1][:, :32]]
    with pytest.raises(ValueError, match="same shape"):
        predict(data)


def test_wrong_dimensions():
    with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
        predict(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        predict([np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)])


def test_empty():
    with pytest.raises(ValueError, match="empty"):
        predict([])
    with pytest.raises(ValueError, match="empty"):
        predict(iter([]))


def test_invalid_batch_size(images):
    with pytest.raises(ValueError, match="batch_size"):
        predict(images, batch_size=0)


def test_invalid_output_settings(images):
    with pytest.raises(ValueError, match="needs an activation"):
        predict(images, output="probs")
    with pytest.raises(ValueError, match="output='probs'"):
        predict(images, activation="softmax", output="logits")
    with pytest.raises(ValueError, match="activation must be"):
        predict(images, activation="relu", output="probs")


def test_custom_class_count(images):
    model = SeededConv(classes=2)
    assert predict(images[:2], model=model).shape == (2, 2, IMAGE_SIZE, IMAGE_SIZE)


def test_callable_model_and_ensemble(images):
    a, b = SeededConv(), SeededConv()
    with torch.no_grad():
        b.conv.weight.mul_(-1)
    ensemble = predict(images, model=[a, b])
    np.testing.assert_allclose(ensemble, 0, atol=1e-5)  # b = -a, so the mean is 0
    doubled = predict(images, model=lambda x: 2 * a(x))
    np.testing.assert_allclose(doubled, 2 * predict(images, model=a), rtol=1e-6)


def test_load_func_error_propagates(images):
    def loader(i):
        if i == 3:
            raise RuntimeError("loader failed")
        return images[i]

    with pytest.raises(RuntimeError, match="loader failed"):
        run_with_timeout(lambda: predict(range(5), load_func=loader, batch_size=1))


def test_model_error_mid_run(images):
    # many small batches keep the prefetch queue full when the error is raised
    data = np.repeat(images, 4, axis=0)
    with pytest.raises(RuntimeError, match="model failed"):
        run_with_timeout(lambda: predict(data, batch_size=1, model=FailOnCall()))


@pytest.mark.parametrize("scenario", ["success", "model_error", "loader_error"])
def test_does_not_leak_threads(images, scenario):
    def loader(i):
        if scenario == "loader_error" and i == 3:
            raise RuntimeError("loader failed")
        return images[i]

    def run():
        model = FailOnCall() if scenario == "model_error" else SeededConv()
        return predict(range(5), batch_size=1, load_func=loader, model=model)

    predict(range(1), load_func=loader)  # long-lived threads (e.g. tqdm's) start now
    baseline = set(threading.enumerate())
    if scenario == "success":
        run_with_timeout(run)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            run_with_timeout(run)
    assert wait_for_new_threads(baseline) == []


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_accelerator_matches_cpu(images, reference, device, dtype):
    result = predict(images, batch_size=2, device=device, dtype=dtype, output="argmax")
    assert (result == reference).mean() >= 0.99


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_accelerator_reduced_precision_outputs_float32(images, device, dtype):
    result = predict(
        images, batch_size=2, device=device, dtype=dtype, activation="softmax", output="probs"
    )
    assert result.dtype == np.float32
    assert np.all((result >= 0) & (result <= 1))


def test_cpu_reduced_precision_warns(images):
    with pytest.warns(UserWarning, match="much slower"):
        predict(images[:1], dtype="bfloat16")


@pytest.mark.parametrize("device", DEVICES)
def test_device_tensor_input(images, device):
    expected = predict(images, batch_size=2, device=device)
    result = predict(torch.from_numpy(images).to(device), batch_size=2, device=device)
    np.testing.assert_array_equal(result, expected)


def run_in_threads(func, calls: int = 8, workers: int = 4) -> list:
    def run_all():
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(func, range(calls)))

    return run_with_timeout(run_all)


@pytest.mark.parametrize("device", ALL_DEVICES)
@pytest.mark.parametrize(
    "input_type", ["numpy", "cpu_tensor", "device_tensor", "device_tensor_fp16"]
)
def test_from_threads(images, device, input_type):
    """Calls from several threads at once must not crash (they did on MPS in OCM) and must
    match a single call."""
    data = {
        "numpy": images,
        "cpu_tensor": torch.from_numpy(images),
        "device_tensor": torch.from_numpy(images).to(device),
        "device_tensor_fp16": torch.from_numpy(images).to(device, torch.float16),
    }[input_type]
    model = SeededConv()
    expected = predict(data, batch_size=2, device=device, model=model)
    results = run_in_threads(lambda _: predict(data, batch_size=2, device=device, model=model))
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_load_func_can_predict(images, device):
    """A load_func may itself run a prediction on the same device (deadlocked on MPS in OCM
    when the lock was held for the whole call)."""
    model = SeededConv()

    def loader(i):
        predict(images[i : i + 1], device=device, model=model)
        return images[i]

    result = run_with_timeout(
        lambda: predict(range(5), batch_size=2, load_func=loader, device=device, model=model)
    )
    np.testing.assert_array_equal(result, predict(images, batch_size=2, device=device, model=model))


def test_a_shared_model_is_cast_once():
    """One thread casts it, with none overlapping, and the others leave it alone. Threads casting
    one model at once failed with dtype errors, or crashed, now and then."""

    class CountsCasts(SeededConv):
        casts, inside, overlapped = 0, 0, False

        def to(self, *args, **kwargs):
            type(self).casts += 1
            type(self).inside += 1
            type(self).overlapped |= type(self).inside > 1
            time.sleep(0.05)
            type(self).inside -= 1
            return super().to(*args, **kwargs)

    model = CountsCasts()
    with pytest.warns(UserWarning, match="much slower"):
        with ThreadPoolExecutor(4) as pool:
            work = [
                pool.submit(predict, np.zeros((1, 3, 8, 8), "f4"), model=model, dtype="bf16")
                for _ in range(4)
            ]
            for future in work:
                future.result()
    assert CountsCasts.casts == 1 and not CountsCasts.overlapped


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_outputs_align_with_inputs(device):
    count, stripe = 12, 4
    result = predict(
        striped_images(count, stripe),
        batch_size=2,
        device=device,
        activation="softmax",
        output="probs",
        nodata=0,
    )
    for i in range(count):
        rows = np.zeros(IMAGE_SIZE, dtype=bool)
        rows[i * stripe : (i + 1) * stripe] = True
        assert np.isnan(result[i][:, rows]).all(), f"image {i} stripe not masked"
        assert np.all(result[i][:, ~rows] > 0), f"image {i} has another's mask"


@pytest.mark.parametrize(
    "make_data, device",
    [
        pytest.param(lambda x: x, "cpu", id="numpy"),
        pytest.param(torch.from_numpy, "cpu", id="cpu_tensor"),
        pytest.param(
            lambda x: torch.from_numpy(x).cuda(),
            "cuda",
            id="cuda_tensor",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            ],
        ),
    ],
)
def test_does_not_modify_input(make_data, device):
    data = make_data(striped_images(5))
    before = data.clone() if isinstance(data, torch.Tensor) else data.copy()
    predict(data, batch_size=2, device=device)
    if isinstance(data, torch.Tensor):
        assert torch.equal(data, before)
    else:
        np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize(
    "dtype, low, high, nodata",
    [
        pytest.param(np.uint8, 1, 255, 0, id="uint8"),
        # values above 32767 catch sign errors if uint16 is reinterpreted as int16
        pytest.param(np.uint16, 30000, 65535, 65535, id="uint16_high"),
        pytest.param(np.int16, -5000, 5000, -9999, id="int16_negative"),
        pytest.param(np.float32, 0, 1, 0, id="float32"),
        pytest.param(np.float64, 0, 1, 0, id="float64"),
    ],
)
def test_input_dtypes_match_float32(dtype, low, high, nodata):
    rng = np.random.default_rng(1)
    shape = (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(low, high, size=shape).astype(dtype)
    else:
        data = rng.uniform(low, high, size=shape).astype(dtype) + 0.01
    data[0, :, :8, :] = nodata
    result = predict(data, batch_size=2, nodata=nodata)
    expected = predict(data.astype(np.float32), batch_size=2, nodata=nodata)
    np.testing.assert_array_equal(result, expected)
    assert np.isnan(result[0, :, :8]).all(), "nodata not detected"


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_uint16_round_trip_is_exact(device):
    """uint16 values travel as int16 and are restored exactly on the device."""
    data = np.array([0, 1, 32767, 32768, 40000, 65535], dtype=np.uint16)
    data = np.broadcast_to(data[None, None, None, :], (2, 1, 1, 6)).copy()
    result = predict(
        data, model=torch.nn.Identity(), nodata=None, normalize=None, device=device, dtype="fp32"
    )
    np.testing.assert_array_equal(result, data.astype(np.float32))


def _non_contiguous_cases() -> dict:
    rng = np.random.default_rng(2)
    big = rng.random((5, 3, 2 * IMAGE_SIZE, 2 * IMAGE_SIZE)).astype(np.float32)
    small = big[:, :, :IMAGE_SIZE, :IMAGE_SIZE].copy()
    return {
        "strided": big[:, :, ::2, ::2],
        "slice_of_larger": big[1:4, :, 10 : 10 + IMAGE_SIZE, 20 : 20 + IMAGE_SIZE],
        "negative_stride": small[:, :, ::-1, :],
        "fortran_order": np.asfortranarray(small),
        "transposed_tensor": torch.from_numpy(small).transpose(2, 3),
    }


@pytest.mark.parametrize("case", list(_non_contiguous_cases()))
def test_non_contiguous_inputs(case):
    data = _non_contiguous_cases()[case]
    contiguous = data.contiguous() if isinstance(data, torch.Tensor) else np.ascontiguousarray(data)
    np.testing.assert_array_equal(predict(data, batch_size=2), predict(contiguous, batch_size=2))


@pytest.mark.parametrize("device", ALL_DEVICES)
@pytest.mark.parametrize("use_generator", [False, True])
def test_results_own_their_memory(images, device, use_generator):
    def run(data):
        return predict((i for i in data) if use_generator else data, batch_size=2, device=device)

    first = run(images)
    snapshot = first.copy()
    run(images[::-1].copy())  # a second run must not overwrite the first result
    np.testing.assert_array_equal(first, snapshot)
    assert first.flags.owndata and first.flags.writeable


def test_generator_lookahead_is_bounded():
    batch_size, count = 2, 40
    state = {"pulled": 0, "processed": 0, "max_lookahead": 0}

    def counting_generator():
        for image in striped_images(count, stripe=1):
            state["pulled"] += 1
            yield image

    class SlowCountingModel(SeededConv):
        def forward(self, x):
            state["max_lookahead"] = max(
                state["max_lookahead"], state["pulled"] - state["processed"]
            )
            time.sleep(0.05)  # slower than loading, so loading runs ahead
            state["processed"] += x.shape[0]
            return super().forward(x)

    predict(counting_generator(), batch_size=batch_size, model=SlowCountingModel())
    assert state["processed"] == count
    assert state["max_lookahead"] <= 8 * batch_size + 1, state["max_lookahead"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("scenario", ["success", "model_error"])
def test_releases_cuda_memory(scenario):
    data = striped_images(8)
    model = SeededConv()
    predict(data, batch_size=2, device="cuda", model=model)  # warm up
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    if scenario == "success":
        predict(data, batch_size=2, device="cuda", model=model)
    else:
        failing = FailOnCall().cuda()
        with pytest.raises(RuntimeError, match="model failed"):
            run_with_timeout(lambda: predict(data, batch_size=2, device="cuda", model=failing))
        del failing
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline


def test_compiled_model_sees_one_shape(monkeypatch):
    """Warm-up on the first batch's shape; the short last batch is padded to it (static
    shapes, no recompiles) and its outputs cut back."""
    height, width, batch_size = 56, 80, 3
    shapes = []

    class RecordingModel(SeededConv):
        def forward(self, x):
            shapes.append(tuple(x.shape))
            return super().forward(x)

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    data = np.random.default_rng(5).random((4, 3, height, width)).astype(np.float32)
    out = predict(data, batch_size=batch_size, compile=True, model=RecordingModel())
    assert shapes == [(batch_size, 3, height, width)] * 3  # warm-up, full batch, padded batch
    np.testing.assert_allclose(out, predict(data, model=SeededConv()), rtol=1e-6)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_collate_fp16_mps_views_to_cpu(images):
    """Moving and casting fp16 MPS views to the CPU in one call returns the first row's
    values for every row (PyTorch 2.11), so collation keeps the steps separate."""
    rows = list(torch.from_numpy(images).to("mps", torch.float16))
    result = to_float32(*collate(rows, torch.device("cpu")))
    np.testing.assert_array_equal(result.numpy(), images.astype(np.float16).astype(np.float32))


def _single_image_batches(count):
    for i in range(count):
        yield RawBatch({"": [np.full((3, 4, 4), i, dtype=np.float32)]}, [None], [i])


def test_prefetch_slow_consumer_keeps_order():
    received = []
    for batch in prefetch(_single_image_batches(6), torch.device("cpu"), depth=1):
        time.sleep(0.25)  # slower than the producer, so it retries puts on a full queue
        assert batch.inputs[""].shape == (1, 3, 4, 4)
        assert batch.inputs[""][0, 0, 0, 0] == batch.items[0]
        assert not batch.is_uint16[""]
        received.append(batch.items[0])
    assert received == list(range(6))


def test_prefetch_close_stops_producer():
    source_closed = threading.Event()

    def endless_batches():
        try:
            for i in itertools.count():
                yield RawBatch({"": [np.zeros((3, 4, 4), dtype=np.float32)]}, [None], [i])
        finally:
            source_closed.set()

    threads_before = threading.active_count()
    prefetched = prefetch(endless_batches(), torch.device("cpu"), depth=1)
    next(prefetched)
    prefetched.close()
    assert source_closed.is_set()
    assert threading.active_count() == threads_before


def test_prefetch_producer_error_propagates():
    def failing_batches():
        yield RawBatch({"": [np.zeros((3, 4, 4), dtype=np.float32)]}, [None], [0])
        raise RuntimeError("load failed")

    prefetched = prefetch(failing_batches(), torch.device("cpu"))
    next(prefetched)
    with pytest.raises(RuntimeError, match="load failed"):
        next(prefetched)


@pytest.mark.parametrize(
    "model, match",
    [
        pytest.param(lambda x: torch.tensor(1.0), r"\(B, C\)", id="scalar"),
        pytest.param(lambda x: x[0], r"\(B, C\)", id="no_batch_dim"),
        pytest.param(lambda x: x[:1], "batch of 2", id="wrong_batch"),
        pytest.param(lambda x: {"mask": np.zeros(3)}, "must be tensors", id="dict_of_numpy"),
        pytest.param(lambda x: {}, "empty dict", id="empty_dict"),
        pytest.param(lambda x: [x], "tensor or a non-empty dict", id="list"),
    ],
)
def test_malformed_model_outputs_raise(images, model, match):
    with pytest.raises((ValueError, TypeError), match=match):
        predict(images[:2], batch_size=2, model=model)


def test_output_size_mismatch_explains(images):
    with pytest.raises(ValueError, match="ceil"):
        predict(images[:2], model=lambda x: x[:, :, :5, :7])
