"""mosaic_dtype: the dtype tiled predictions are blended in."""

import numpy as np
import pytest
import torch
from helpers import SeededConv

from manypred import predict_array

KW = {
    "patch_size": 64,
    "overlap": 16,
    "progress": False,
    "normalize": {"type": "zscore", "scope": "tile"},
}


def scene(size=200):
    return np.random.default_rng(0).random((3, size, size)).astype(np.float32) + 0.1


@pytest.mark.parametrize("mosaic_dtype, atol", [("float16", 5e-3), ("bfloat16", 3e-2)])
def test_half_precision_mosaic_is_close_to_float32(mosaic_dtype, atol):
    x = scene()
    full = predict_array(SeededConv(), x, device="cpu", **KW)
    half = predict_array(SeededConv(), x, device="cpu", mosaic_dtype=mosaic_dtype, **KW)
    assert half.dtype == np.float32  # outputs stay float32
    np.testing.assert_allclose(half, full, atol=atol, rtol=atol)
    classes = predict_array(
        SeededConv(), x, device="cpu", mosaic_dtype=mosaic_dtype, output="argmax", **KW
    )
    reference = predict_array(SeededConv(), x, device="cpu", output="argmax", **KW)
    assert (classes == reference).mean() > 0.99


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_mosaic_uses_less_memory_on_the_gpu():
    x = scene(4096)
    peaks = {}
    for mosaic_dtype in ("float16", "float32"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = predict_array(
            SeededConv(),
            x,
            device="cuda",
            dtype="float16",
            mosaic_dtype=mosaic_dtype,
            batch_size=4,
            **KW,
        )
        torch.cuda.synchronize()
        peaks[mosaic_dtype] = torch.cuda.max_memory_allocated()
        assert out.dtype == np.float32
    # a 4-class 4096 x 4096 mosaic plus its weights: 336 MB in float32, 168 MB in float16; the
    # scene is this large so the mosaic outweighs the batches, which are the same in both
    assert peaks["float32"] - peaks["float16"] > 30e6, peaks


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_blending_matches_ocm_more_closely():
    """OCM blends in the inference dtype, so float16 inference matches it with a float16 mosaic."""
    ocm = pytest.importorskip("omnicloudmask")
    import goldens
    from ocm_compat import OCM_NORMALIZE, ocm_confidence

    image = goldens.make_image(
        goldens.load_inputs(),
        {"source": "s2", "rows": [0, 300], "cols": [0, 380], "dtype": "uint16"},
    )
    expected = ocm.predict_from_array(
        image,
        patch_size=128,
        patch_overlap=40,
        inference_device="cuda",
        inference_dtype="float16",
        export_confidence=True,
        custom_models=goldens.load_tiny("tiny0"),
    )
    errors = {}
    for mosaic_dtype in ("float16", "float32"):
        pred = predict_array(
            goldens.load_tiny("tiny0"),
            image,
            patch_size=128,
            overlap=40,
            device="cuda",
            dtype="float16",
            mosaic_dtype=mosaic_dtype,
            normalize=OCM_NORMALIZE,
            output=ocm_confidence,
            progress=False,
        )
        errors[mosaic_dtype] = float(np.abs(pred - expected).max())
    assert errors["float16"] <= errors["float32"], errors
    assert errors["float16"] < 5e-3
