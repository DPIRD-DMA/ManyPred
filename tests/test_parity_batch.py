"""predict_batch vs OmniCloudMask's predict_from_batch goldens.

On the workstation that generated the goldens the CPU results are bitwise equal; other CPUs
(CI runners) round differently by up to ~1e-6, so floats are compared within 1e-5 and classes
exactly except at near-ties."""

import goldens
import numpy as np
import pytest
import torch
from ocm_compat import manypred_kwargs, missing_as_nan

from manypred import predict_batch

CASES = [n for n in goldens.case_names() if n.startswith("batch_")]
TINY = [n for n in CASES if not n.endswith(goldens.OCM_MODELS)]
REAL = [n for n in CASES if n.endswith(goldens.OCM_MODELS)]


def models_for(name: str):
    if name == goldens.OCM_MODELS:
        return goldens.load_ocm_models()
    return [goldens.load_tiny(part) for part in name.split("+")]


def run(name: str, device: str = "cpu", dtype: str = "float32"):
    case = goldens.load_case(name)
    meta = case["meta"]
    chips = goldens.make_chips(goldens.load_inputs(), meta["input"])
    kwargs = {**manypred_kwargs(meta), "dtype": dtype}
    pred = predict_batch(models_for(meta["model"]), chips, device=device, **kwargs)
    return pred, case


def assert_parity(pred: np.ndarray, case: dict, exact: bool, atol: float = 0.0):
    golden = case["output"]
    assert pred.shape == golden.shape
    assert pred.dtype == golden.dtype
    if golden.dtype == np.uint8:
        differ = np.flatnonzero(pred.ravel() != golden.ravel())
        allowed = np.setdiff1d(differ, case["near_tie"])
        assert len(allowed) == 0, f"{len(allowed)} classes differ away from near-ties"
        if exact:
            assert len(differ) == 0, f"{len(differ)} near-tie classes differ"
    elif exact:
        np.testing.assert_array_equal(pred, missing_as_nan(golden, pred))
    else:
        golden = missing_as_nan(golden, pred)
        np.testing.assert_allclose(pred, golden, rtol=0, atol=atol, equal_nan=True)


@pytest.mark.parametrize("name", TINY)
def test_batch_parity_cpu(name):
    pred, case = run(name)
    # bitwise on the workstation that made the goldens; other CPUs round differently (~1e-6)
    assert_parity(pred, case, exact=False, atol=1e-5)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", REAL)
def test_batch_parity_cpu_ocm_models(name):
    pred, case = run(name)
    # bitwise on the workstation that made the goldens; other CPUs round differently (~1e-6)
    assert_parity(pred, case, exact=False, atol=1e-5)


@pytest.fixture
def no_tf32(monkeypatch):
    """cuDNN convolutions use TF32 by default, which alone moves logits by ~2e-3; without it
    CUDA matches the CPU goldens to ~5e-6."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", TINY)
def test_batch_parity_cuda(name, no_tf32):
    pred, case = run(name, device="cuda")
    assert_parity(pred, case, exact=False, atol=1e-4)
