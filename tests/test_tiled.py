"""predict_array with synthetic models whose exact outputs are known."""

import warnings

import numpy as np
import pytest
import torch
from helpers import ALL_DEVICES, DEVICES, SeededConv, random_images

from manypred import predict_array, predict_batch
from manypred.blend import gradient_weights
from manypred.windows import (
    adjust_patch_size,
    make_patch_indexes,
    nodata_pixels,
    plan_windows,
    shift_window,
)

KW = {"progress": False, "device": "cpu"}
SIZES = [
    (64, 64, 64, 0),  # one patch
    (100, 130, 32, 0),
    (100, 130, 32, 8),
    (97, 211, 40, 13),  # nothing divides
    (50, 50, 20, 10),  # overlap = half the patch
    (70, 300, 64, 63),  # stride 1 along the long side
]


def coordinates(height, width, offset=1.0):
    """(3, H, W): row + offset, column + offset, constant. Nodata (0) never occurs."""
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([rows + offset, cols + offset, np.full_like(rows, 7.0)])


@pytest.mark.parametrize("height, width, patch, overlap", SIZES)
def test_identity_model_returns_input(height, width, patch, overlap):
    image = coordinates(height, width)
    out = predict_array(
        torch.nn.Identity(), image, patch_size=patch, overlap=overlap, nodata=None, **KW
    )
    np.testing.assert_allclose(out, image, rtol=1e-5)


@pytest.mark.parametrize("height, width, patch, overlap", SIZES)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_constant_model_gives_constant_mosaic(height, width, patch, overlap, batch_size):
    def constant(x):
        return torch.full((x.shape[0], 2, *x.shape[2:]), 0.375)

    image = random_images(1, size=max(height, width))[0][:, :height, :width]
    out = predict_array(
        constant, image, patch_size=patch, overlap=overlap, batch_size=batch_size, **KW
    )
    np.testing.assert_allclose(out, 0.375, rtol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_shifted_patches_map_back_to_their_source(seed):
    """With nodata blobs, patches are skipped and shifted; an identity model on coordinate
    channels must still return every valid pixel's own coordinates, and cover every valid
    pixel."""
    rng = np.random.default_rng(seed)
    height, width = rng.integers(80, 160, size=2)
    image = coordinates(height, width)
    for _ in range(rng.integers(1, 5)):
        top, left = rng.integers(0, height - 10), rng.integers(0, width - 10)
        image[:, top : top + rng.integers(10, 60), left : left + rng.integers(10, 60)] = 0
    image[:, :, : rng.integers(0, 40)] = 0  # a nodata edge
    valid = ~(image == 0).all(axis=0)
    out = predict_array(torch.nn.Identity(), image, patch_size=32, overlap=8, nodata=0, **KW)
    np.testing.assert_allclose(out[:, valid], image[:, valid], rtol=1e-5)
    assert np.isnan(out[:, ~valid]).all()


def test_all_nodata_scene():
    image = np.zeros((3, 64, 80), dtype=np.float32)
    logits = predict_array(SeededConv(), image, patch_size=32, overlap=8, **{**KW, "nodata": 0})
    assert logits.shape == (4, 64, 80) and np.isnan(logits).all()
    classes = predict_array(
        SeededConv(), image, patch_size=32, overlap=8, output="argmax", **{**KW, "nodata": 0}
    )
    assert classes.shape == (1, 64, 80) and np.all(classes == 0)


def test_matches_batch_mode_for_one_patch():
    image = random_images(1)[0]
    image[:, :8] = 0
    kwargs = {"normalize": {"type": "zscore", "scope": "tile"}, "progress": False, "device": "cpu"}
    tiled = predict_array(SeededConv(), image, patch_size=64, overlap=0, **kwargs)
    batch = predict_batch(SeededConv(), image[None], **kwargs)
    np.testing.assert_array_equal(tiled, batch[0])


def test_probs_average_probabilities():
    """Activation is per patch, before blending: where patches overlap, probs are the weighted
    mean of each patch's softmax, which differs from the softmax of the blended logits.
    Per-patch z-scoring makes the identity model's logits differ between patches."""
    image = coordinates(40, 64)
    kwargs = {
        **KW,
        "patch_size": 40,
        "overlap": 16,
        "nodata": None,
        "normalize": {"type": "zscore", "scope": "tile"},
    }
    probs = predict_array(
        torch.nn.Identity(), image, activation="softmax", output="probs", **kwargs
    )
    after = predict_array(
        torch.nn.Identity(), image, output=lambda x: torch.softmax(x, dim=1), **kwargs
    )
    np.testing.assert_allclose(probs.sum(axis=0), 1, rtol=1e-5)
    # columns 0-23 are in the first patch only, 40-63 in the last only, 24-39 in both
    np.testing.assert_allclose(probs[:, :, :24], after[:, :, :24], rtol=1e-5)
    assert np.abs(probs[:, :, 24:40] - after[:, :, 24:40]).max() > 1e-2


def test_all_nodata_scene_checks_outputs():
    image = np.zeros((3, 64, 80), dtype=np.float32)
    with pytest.raises(ValueError, match="ceil"):
        predict_array(
            lambda x: x[:, :, :5, :7], image, patch_size=32, overlap=8, **{**KW, "nodata": 0}
        )
    with pytest.raises(ValueError, match="image-level"):
        predict_array(
            lambda x: x.mean(dim=(2, 3)), image, patch_size=32, overlap=8, **{**KW, "nodata": 0}
        )
    tiles = predict_array(
        lambda x: x.mean(dim=(2, 3)),
        image,
        patch_size=32,
        overlap=8,
        output="logits",
        **{**KW, "nodata": 0},
    )
    assert tiles.values.shape == (0, 3)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tensor_with_cpu_inference():
    image = coordinates(90, 120)
    out = predict_array(
        torch.nn.Identity(),
        torch.from_numpy(image).cuda(),
        patch_size=32,
        overlap=8,
        nodata=None,
        progress=False,
        device="cpu",
    )
    np.testing.assert_allclose(out, image, rtol=1e-5)


@pytest.mark.parametrize("device", ALL_DEVICES)
@pytest.mark.parametrize("as_tensor", [False, True])
def test_devices_and_tensor_inputs(device, as_tensor):
    image = coordinates(90, 120)
    data = torch.from_numpy(image).to(device) if as_tensor else image
    out = predict_array(
        torch.nn.Identity(),
        data,
        patch_size=32,
        overlap=8,
        nodata=None,
        progress=False,
        device=device,
        dtype="float32",
    )
    np.testing.assert_allclose(out, image, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_cpu_mosaic_with_accelerator(device):
    image = coordinates(90, 120)
    out = predict_array(
        torch.nn.Identity(),
        image,
        patch_size=32,
        overlap=8,
        nodata=None,
        progress=False,
        device=device,
        mosaic_device="cpu",
    )
    np.testing.assert_allclose(out, image, rtol=1e-5)


def test_scene_smaller_than_patch_warns_and_shrinks():
    image = coordinates(40, 90)
    with pytest.warns(UserWarning, match="larger than the scene"):
        out = predict_array(
            torch.nn.Identity(), image, patch_size=64, overlap=20, nodata=None, **KW
        )
    np.testing.assert_allclose(out, image, rtol=1e-5)


def test_wrong_image_shape():
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        predict_array(torch.nn.Identity(), np.zeros((64, 64)), patch_size=32, **KW)


def test_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        predict_array(torch.nn.Identity(), coordinates(64, 64), patch_size=32, overlap=32, **KW)


# --- windows and weights -----------------------------------------------------------------


@pytest.mark.parametrize("height, width, patch, overlap", SIZES)
def test_patch_indexes_cover_scene(height, width, patch, overlap):
    windows = make_patch_indexes(height, width, patch, overlap)
    covered = np.zeros((height, width), dtype=int)
    for top, bottom, left, right in windows:
        assert bottom - top == patch and right - left == patch
        assert 0 <= top and bottom <= height and 0 <= left and right <= width
        covered[top:bottom, left:right] += 1
    assert covered.min() >= 1


def test_shift_window_moves_off_nodata_edges():
    nodata = np.zeros((20, 20), dtype=bool)
    nodata[:5] = True  # top rows
    nodata[:, 17:] = True  # right columns
    assert shift_window(nodata, (0, 10, 0, 10)) == (5, 15, 0, 10)
    assert shift_window(nodata, (5, 15, 10, 20)) == (5, 15, 7, 17)
    # never leaves the scene
    nodata[:] = False
    nodata[15:] = True
    assert shift_window(nodata, (10, 20, 0, 10)) == (5, 15, 0, 10)


def test_plan_windows_skips_and_dedupes():
    nodata = np.zeros((40, 40), dtype=bool)
    nodata[:, :20] = True
    windows = plan_windows(nodata, 20, 0)
    # the left patches are skipped, the right ones kept
    assert windows == [(0, 20, 20, 40), (20, 40, 20, 40)]


def test_nodata_pixels_nan_aware():
    image = np.ones((2, 3, 3), dtype=np.float32)
    image[:, 0, 0] = np.nan
    image[0, 1, 1] = np.nan  # one band only: not nodata
    image[:, 2, 2] = 0
    assert nodata_pixels(image, np.nan).sum() == 1
    assert nodata_pixels(image, 0).tolist() == [
        [True, False, False],
        [False, False, False],
        [False, False, True],
    ]
    torch_result = nodata_pixels(torch.from_numpy(image), 0)
    np.testing.assert_array_equal(torch_result, nodata_pixels(image, 0))


def test_gradient_weights_match_ocm_formula():
    patch, overlap = 10, 3
    w = gradient_weights(patch, overlap).numpy()
    ramp = np.full(patch, overlap, dtype=np.float32)
    ramp[:overlap] = np.arange(1, overlap + 1)
    ramp[-overlap:] = np.arange(overlap, 0, -1)
    ramp /= overlap
    np.testing.assert_allclose(w, np.outer(ramp, ramp), rtol=1e-5)
    assert w.max() == 1 and w.min() == pytest.approx(1 / 9)
    assert np.all(gradient_weights(8, 0).numpy() == 1)
    # overlap above half the patch is capped at half
    np.testing.assert_array_equal(gradient_weights(8, 6).numpy(), gradient_weights(8, 4).numpy())


def test_adjust_patch_size():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert adjust_patch_size(500, 600, 0.0, 256, 64) == (256, 64)
    with pytest.warns(UserWarning, match="nodata"):
        assert adjust_patch_size(500, 600, 0.4, 300, 200) == (250, 125)
    with pytest.warns(UserWarning, match="larger than the scene"):
        assert adjust_patch_size(100, 600, 0.0, 256, 64) == (100, 50)


def test_patch_size_is_not_adapted_by_default():
    image = coordinates(200, 240)
    image[:, :, :150] = 0  # over 30 % nodata, patch over half the scene
    sizes = []

    def model(x):
        sizes.append(x.shape[-1])
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict_array(model, image, patch_size=128, overlap=32, **{**KW, "nodata": 0})
    assert set(sizes) == {128}
    sizes.clear()
    with pytest.warns(UserWarning, match="30%"):
        predict_array(
            model, image, patch_size=128, overlap=32, adapt_patch_size=True, **{**KW, "nodata": 0}
        )
    assert set(sizes) == {100}


def test_shift_patches_off_keeps_the_plain_grid():
    image = coordinates(96, 96)
    image[:, :20] = 0  # nodata rows at the top
    tops = []

    def model(x):
        tops.append(float(x[0, 0, 0, 0]))  # row coordinate + 1 of the patch's first row
        return x

    predict_array(
        model,
        image,
        patch_size=32,
        overlap=0,
        batch_size=1,
        shift_patches=False,
        **{**KW, "nodata": 0},
    )
    shifted = []

    def model_shifted(x):
        shifted.append(float(x[0, 0, 0, 0]))
        return x

    predict_array(
        model_shifted, image, patch_size=32, overlap=0, batch_size=1, **{**KW, "nodata": 0}
    )
    # the top row of patches starts at row 0 (all nodata there) without shifting, and is
    # moved down to the first valid row (21) with it
    assert 0.0 in tops and 0.0 not in shifted and 21.0 in shifted


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_mosaics_never_share_cached_weights(device):
    """Regression: blend weights copied to the prediction device were cached by id(), which
    Python reuses, so a later mosaic could blend with an earlier one's weights (same shape,
    other overlap). Alternate overlaps with the mosaic on the CPU."""
    image = coordinates(90, 120)
    for _ in range(3):
        for overlap in (0, 8, 16):
            out = predict_array(
                torch.nn.Identity(),
                image,
                patch_size=32,
                overlap=overlap,
                nodata=None,
                progress=False,
                device=device,
                mosaic_device="cpu",
            )
            np.testing.assert_allclose(out, image, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_nearest_upsampling_on_cuda():
    """torch 2.14's Inductor with dynamic shapes miscomputes nearest upsampling on CUDA at some
    sizes (132 px patches here gave errors of 0.5); ManyPred compiles with static shapes."""
    import torch.nn.functional as F

    class Down(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 3, padding=1)

        def forward(self, x):
            low = F.avg_pool2d(self.conv(x), 2, ceil_mode=True)
            return F.interpolate(low, size=x.shape[-2:], mode="nearest")

    torch.manual_seed(0)
    model = Down()
    scene = np.random.default_rng(0).random((3, 300, 260)).astype(np.float32)
    kw = {**KW, "device": "cuda", "patch_size": 132, "overlap": 16, "batch_size": 3}
    kw["dtype"] = "float32"  # the bug is in the compiled indexing, whatever the dtype
    eager = predict_array(model, scene, **kw)
    compiled = predict_array(model, scene, compile=True, **kw)
    np.testing.assert_allclose(compiled, eager, atol=1e-5)
