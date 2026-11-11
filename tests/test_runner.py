"""The runner: ensembles, compiling, loader shutdown and models shared between threads."""

import threading
import time
import warnings

import numpy as np
import pytest
import torch

from manypred import predict_array
from manypred import runner as runner_module
from manypred.runner import RawBatch, _claimed, ensemble, prefetch

KW = {"device": "cpu", "progress": False}


def test_an_ensemble_may_return_its_outputs_in_any_order():
    x = torch.ones(1, 1, 2, 2)
    out = ensemble([lambda x: {"a": x, "b": 2 * x}, lambda x: {"b": 4 * x, "a": 3 * x}], x)
    assert list(out) == ["a", "b"]
    assert out["a"].mean() == 2 and out["b"].mean() == 3


def test_an_ensemble_with_different_outputs_is_refused():
    with pytest.raises(ValueError, match="same outputs"):
        ensemble([lambda x: {"a": x}, lambda x: {"b": x}], torch.ones(1, 1, 2, 2))


def test_a_scene_without_data_compiles_for_the_scenes_batch_size(monkeypatch):
    """An all-nodata scene learns its outputs from one patch; compiling there used to fix the
    batch size at 1, so every later batch recompiled."""
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    shapes = []

    def model(x):
        shapes.append(x.shape[0])
        return x

    image = np.zeros((1, 64, 64), np.float32)
    predict_array(
        model, image, patch_size=32, overlap=8, nodata=0, compile=True, batch_size=4, **KW
    )
    assert shapes and set(shapes) == {4}


def test_closing_a_prefetch_doesnt_wait_forever_for_a_stuck_loader(monkeypatch):
    """After an error in the caller, closing waited for a loader blocked in a read."""
    monkeypatch.setattr(runner_module, "_STOP_TIMEOUT", 0.2)
    release = threading.Event()

    def batches():
        yield RawBatch({"": [np.zeros((1, 4, 4), np.float32)]}, [None], [None])
        release.wait()  # a read that never returns
        yield RawBatch({"": [np.zeros((1, 4, 4), np.float32)]}, [None], [None])

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="model failed"):
        for _ in prefetch(batches(), torch.device("cpu")):
            raise RuntimeError("model failed")
    assert time.perf_counter() - start < 3
    release.set()


def test_a_model_in_use_elsewhere_isnt_recast():
    """Two threads sharing a module in different dtypes used to recast it mid-inference."""
    model = torch.nn.Conv2d(1, 1, 1)
    image = np.ones((1, 32, 32), np.float32)
    with _claimed([model], torch.device("cpu"), torch.bfloat16):
        with pytest.raises(RuntimeError, match="being run on cpu in bfloat16"):
            predict_array(model, image, patch_size=32, overlap=0, **KW)
        assert model.weight.dtype == torch.float32  # untouched
    # the same device and dtype may share it, and the claim is released afterwards
    with _claimed([model], torch.device("cpu"), torch.float32):
        predict_array(model, image, patch_size=32, overlap=0, **KW)
    assert not runner_module._IN_USE


def test_threads_sharing_a_model_in_different_dtypes():
    model = torch.nn.Conv2d(1, 1, 1)
    started, release = threading.Event(), threading.Event()
    slow_forward = model.forward

    def forward(x):
        started.set()
        release.wait(5)
        return slow_forward(x)

    model.forward = forward
    image = np.ones((1, 32, 32), np.float32)
    errors = []
    thread = threading.Thread(
        target=lambda: predict_array(model, image, patch_size=32, overlap=0, **KW)
    )
    thread.start()
    started.wait(5)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # bfloat16 on the CPU
            predict_array(model, image, patch_size=32, overlap=0, dtype="bfloat16", **KW)
    except RuntimeError as e:
        errors.append(e)
    finally:
        release.set()
        thread.join()
    assert errors and "another call" in str(errors[0])
    assert model.weight.dtype == torch.float32


def test_a_submodule_shared_by_two_models_isnt_recast():
    """Claims covered only the top-level module, so another wrapper around the same conv
    could recast it under the first run."""
    shared = torch.nn.Conv2d(1, 1, 1)
    first, second = torch.nn.Sequential(shared), torch.nn.Sequential(shared)
    image = np.ones((1, 32, 32), np.float32)
    with _claimed([first], torch.device("cpu"), torch.float32):
        with pytest.raises(RuntimeError, match="another call"):
            predict_array(second, image, patch_size=32, overlap=0, dtype="float64", **KW)
        assert shared.weight.dtype == torch.float32
    assert not runner_module._IN_USE


def test_a_parameter_shared_by_two_models_isnt_recast():
    weight = torch.nn.Parameter(torch.ones(1, 1, 1, 1))
    first, second = torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.Conv2d(1, 1, 1, bias=False)
    first.weight = second.weight = weight  # tied weights, separate modules
    image = np.ones((1, 32, 32), np.float32)
    with _claimed([first], torch.device("cpu"), torch.float32):
        with pytest.raises(RuntimeError, match="another call"):
            predict_array(second, image, patch_size=32, overlap=0, dtype="float64", **KW)
    assert weight.dtype == torch.float32
