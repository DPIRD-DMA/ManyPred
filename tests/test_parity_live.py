"""ManyPred vs OmniCloudMask called live (needs the ``parity`` dependency group).

In float16/bfloat16 OCM accumulates the mosaic in the inference dtype and ManyPred in float32,
so they are compared with a tolerance.
"""

import goldens
import numpy as np
import pytest
import torch
from ocm_compat import OCM_NORMALIZE, ocm_confidence

from manypred import normalize_numpy, normalize_torch, predict_array, predict_batch

ocm = pytest.importorskip("omnicloudmask")

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def scene():
    inputs = goldens.load_inputs()
    return goldens.make_image(
        inputs, {"source": "s2", "rows": [0, 300], "cols": [0, 380], "dtype": "uint16"}
    )


def test_live_tiled_fp32_cpu(scene):
    kwargs = {"patch_size": 128, "patch_overlap": 40}
    expected = ocm.predict_from_array(
        scene,
        inference_device="cpu",
        export_confidence=True,
        custom_models=goldens.load_tiny("tiny0"),
        **kwargs,
    )
    pred = predict_array(
        goldens.load_tiny("tiny0"),
        scene,
        patch_size=128,
        overlap=40,
        device="cpu",
        normalize=OCM_NORMALIZE,
        output=ocm_confidence,
        progress=False,
    )
    np.testing.assert_allclose(pred, expected, atol=1e-4)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_live_reduced_precision_cuda(scene, dtype):
    tol = {"float16": 5e-3, "bfloat16": 3e-2}[dtype]
    expected = ocm.predict_from_array(
        scene,
        patch_size=128,
        patch_overlap=40,
        batch_size=1,
        inference_device="cuda",
        inference_dtype=dtype,
        export_confidence=True,
        custom_models=goldens.load_tiny("tiny0"),
    )
    pred = predict_array(
        goldens.load_tiny("tiny0"),
        scene,
        patch_size=128,
        overlap=40,
        device="cuda",
        dtype=dtype,
        normalize=OCM_NORMALIZE,
        output=ocm_confidence,
        progress=False,
    )
    np.testing.assert_allclose(pred, expected, atol=tol)
    chips = np.stack([scene[:, :64, :64], scene[:, 64:128, 64:128]])
    expected = ocm.predict_from_batch(
        chips,
        inference_device="cuda",
        inference_dtype=dtype,
        custom_models=goldens.load_tiny("tiny0"),
    )
    pred = predict_batch(
        goldens.load_tiny("tiny0"),
        chips,
        device="cuda",
        dtype=dtype,
        normalize=OCM_NORMALIZE,
        output="argmax",
        progress=False,
    )
    assert (pred == expected).mean() > 0.99


@pytest.mark.parametrize("nodata", [0, float("nan")])
def test_ocm_normalisation_as_a_spec(nodata):
    """OCM's per-patch normalisation is {"type": "zscore", "scope": "tile"}."""
    from omnicloudmask.model_utils import channel_norm, channel_norm_torch

    patch = goldens.make_image(
        goldens.load_inputs(),
        {
            "source": "lc8",
            "rows": [0, 128],
            "cols": [160, 288],
            "dtype": "float32_nan" if nodata != 0 else "float32",
        },
    )
    ours = normalize_numpy(patch, OCM_NORMALIZE, nodata=nodata)
    np.testing.assert_allclose(ours, channel_norm(patch, nodata), atol=1e-5)
    batch = torch.from_numpy(patch)[None]
    np.testing.assert_array_equal(
        normalize_torch(batch, OCM_NORMALIZE, nodata=nodata).numpy(),
        channel_norm_torch(batch, nodata).numpy(),
    )
