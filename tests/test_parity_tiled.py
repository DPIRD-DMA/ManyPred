"""predict_array vs OmniCloudMask's predict_from_array goldens.

OCM normalises patches with numpy on the host and ManyPred with torch on the device, so the
two differ by float rounding: blended values within ``ATOL``, and classes equal except at
near-ties (top-2 logits within 1e-3).
"""

import warnings

import goldens
import numpy as np
import pytest
import torch
from ocm_compat import manypred_kwargs, missing_as_nan

from manypred import predict_array

ATOL = 1e-4

CASES = [n for n in goldens.case_names() if n.startswith("array_")]
# OCM's tiled path does not detect NaN nodata (`patch == nan` is never true), so it neither
# skips nor shifts NaN patches; ManyPred does. Those cases are compared in their own test.
NAN = [n for n in CASES if "_nan_" in n]
TINY = [n for n in CASES if not n.endswith(goldens.OCM_MODELS) and n not in NAN]
REAL = [n for n in CASES if n.endswith(goldens.OCM_MODELS)]


def models_for(name):
    if name == goldens.OCM_MODELS:
        return goldens.load_ocm_models()
    return [goldens.load_tiny(part) for part in name.split("+")]


def run(name, device="cpu", **overrides):
    case = goldens.load_case(name)
    meta = case["meta"]
    image = goldens.make_image(goldens.load_inputs(), meta["input"])
    kwargs = {**manypred_kwargs(meta), **overrides}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = predict_array(models_for(meta["model"]), image, device=device, **kwargs)
    return pred, case, [str(w.message) for w in caught]


def assert_parity(pred, case, atol=ATOL):
    golden = case["output"]
    assert pred.shape == golden.shape and pred.dtype == golden.dtype
    if golden.dtype == np.uint8:
        differ = np.flatnonzero(pred.ravel() != golden.ravel())
        unexplained = np.setdiff1d(differ, case["near_tie"])
        assert len(unexplained) == 0, f"{len(unexplained)} classes differ away from near-ties"
    else:
        golden = missing_as_nan(golden, pred)
        np.testing.assert_array_equal(np.isnan(pred), np.isnan(golden))
        np.testing.assert_allclose(pred, golden, rtol=0, atol=atol, equal_nan=True)


@pytest.mark.parametrize("name", TINY)
def test_tiled_parity_cpu(name):
    pred, case, caught = run(name)
    assert_parity(pred, case)
    # ManyPred warns when OCM did, about patch shrinking (not OCM's small-image warnings)
    expected = [w for w in case["meta"]["warnings"] if "Adjusting" in w or "too large" in w]
    assert len([w for w in caught if "patch_size" in w]) == len(expected)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", REAL)
def test_tiled_parity_cpu_ocm_models(name):
    pred, case, _ = run(name)
    assert_parity(pred, case)


@pytest.fixture
def no_tf32(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", TINY)
@pytest.mark.parametrize("mosaic_device", [None, "cpu"])
def test_tiled_parity_cuda(name, mosaic_device, no_tf32):
    pred, case, _ = run(name, device="cuda", mosaic_device=mosaic_device)
    assert_parity(pred, case)


@pytest.mark.parametrize("name", NAN)
def test_tiled_nan_nodata_matches_ocm_away_from_nodata(name):
    """With NaN nodata OCM predicts all-NaN patches (NaN normalised to 0) and never shifts
    patches away from NaN edges; ManyPred skips and shifts them as for any nodata value. More
    than a patch away from any nodata pixel both blend the same patches, so they must agree."""
    pred, case, _ = run(name)
    meta = case["meta"]
    patch = meta["kwargs"]["patch_size"]
    image = goldens.make_image(goldens.load_inputs(), meta["input"])
    nodata = torch.from_numpy(np.isnan(image).all(axis=0)).float()[None, None]
    near = torch.nn.functional.max_pool2d(nodata, 4 * patch + 1, stride=1, padding=2 * patch)
    far = ~near[0, 0].bool().numpy()
    assert far.sum() > 1000, "test needs pixels away from nodata"
    golden = case["output"]
    if golden.dtype == np.uint8:
        tie = np.zeros(golden.size, dtype=bool)
        tie[case["near_tie"]] = True
        differ = (pred != golden).ravel() & ~tie
        assert not differ.reshape(golden.shape)[:, far].any()
    else:
        np.testing.assert_allclose(pred[:, far], golden[:, far], rtol=0, atol=ATOL)
    # nodata is class 0, or NaN in float outputs; valid pixels have values
    missing = nodata[0, 0].bool().numpy()
    at_nodata = pred[:, missing]
    assert (at_nodata == 0).all() if golden.dtype == np.uint8 else np.isnan(at_nodata).all()
    assert not np.isnan(pred[:, ~missing]).any()
