"""merge="cut": each pixel from the patch with the largest gradient weight there."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import SeededConv

from manypred import predict_array, predict_files
from manypred.blend import gradient_weights
from manypred.lazy import ArrayReader
from manypred.outputs import make_specs
from manypred.runner import Runner
from manypred.tiled import Tiler, pad_value, read_patch
from manypred.windows import nodata_pixels, plan_windows

KW = {"device": "cpu", "progress": False}
RNG = np.random.default_rng(0)


def offset(x):
    """Each patch's values plus the patch's mean: the same pixel differs between patches."""
    return x[:, :2] + x.mean(dim=(1, 2, 3), keepdim=True)


def corner(x):
    """Every pixel of a patch holds the patch's top-left pixel (of a coordinates image: the
    window's top + 1 and left + 1), so the output says which patch each pixel came from."""
    return x[:, :2, :1, :1].expand(-1, -1, *x.shape[2:]).clone()


class Heads(torch.nn.Module):
    """A full-resolution and a quarter-resolution head, both patch-dependent, and an
    image-level one."""

    def forward(self, x):
        dense = offset(x)
        return {"full": dense, "quarter": F.avg_pool2d(dense, 4), "score": x.mean(dim=(2, 3))}


HEAD_KW = {"activation": {"score": "sigmoid"}, "output": {"score": "probs"}}


def coordinates(height, width):
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([rows + 1, cols + 1, np.full_like(rows, 7.0)])


def scene(height, width, holes=True):
    x = RNG.random((3, height, width)).astype(np.float32) + 0.1
    if holes:
        x[:, : height // 5, : width // 3] = 0
        x[:, -height // 3 :, -width // 4 :] = 0
    return x


def cut_reference(
    model,
    image,
    patch,
    overlap,
    nodata=None,
    factor=1,
    shift=True,
    mask=True,
    dtype=np.float32,
    multiple=None,
):
    """The cut mosaic computed directly: every planned patch predicted on its own, its weight
    map painted over the scene (0 elsewhere), and per pixel the prediction of the first patch
    with the largest weight (np.argmax). NaN where no patch covered; masked like ManyPred."""
    missing = nodata_pixels(image, nodata)
    windows = plan_windows(missing, patch, overlap, shift=shift, multiple=multiple or factor)
    height, width = -(-image.shape[1] // factor), -(-image.shape[2] // factor)
    weights = gradient_weights(patch // factor, overlap // factor).numpy().astype(dtype)
    pad = pad_value(image, nodata)
    weight_maps, value_maps = [], []
    for top, bottom, left, right in windows:
        crop = read_patch(image, (top, bottom, left, right), pad)
        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(crop))[None])[0].numpy()
        out = out.astype(dtype).astype(np.float32)
        t, lft = top // factor, left // factor
        rows, cols = min(out.shape[1], height - t), min(out.shape[2], width - lft)
        w = np.zeros((height, width), dtype=np.float64)
        v = np.zeros((out.shape[0], height, width), dtype=np.float32)
        w[t : t + rows, lft : lft + cols] = weights[:rows, :cols]
        v[:, t : t + rows, lft : lft + cols] = out[:, :rows, :cols]
        weight_maps.append(w)
        value_maps.append(v)
    weight_maps, value_maps = np.stack(weight_maps), np.stack(value_maps)
    winner = np.argmax(weight_maps, axis=0)  # the first patch on ties
    result = np.take_along_axis(value_maps, winner[None, None], axis=0)[0]
    result[:, weight_maps.max(axis=0) == 0] = np.nan
    if mask and nodata is not None:
        valid = ~missing
        if factor > 1:
            padded = np.zeros((height * factor, width * factor), dtype=bool)
            padded[: valid.shape[0], : valid.shape[1]] = valid
            valid = padded.reshape(height, factor, width, factor).any(axis=(1, 3))
        result[:, ~valid] = np.nan
    return result, windows


@pytest.mark.parametrize(
    "height, width, patch, overlap",
    [
        (64, 64, 64, 0),  # one patch
        (100, 130, 32, 8),
        (97, 211, 40, 13),  # nothing divides: clamped last row and column
        (50, 50, 20, 10),  # overlap = half the patch
        (70, 300, 64, 63),  # stride 1 along the long side
        (90, 90, 30, 0),  # no overlap: weights all 1, the first patch keeps the clamped part
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cut_equals_the_argmax_of_the_weights(height, width, patch, overlap, batch_size):
    image = scene(height, width, holes=False)
    expected, _ = cut_reference(offset, image, patch, overlap)
    out = predict_array(
        offset, image, patch_size=patch, overlap=overlap, batch_size=batch_size, merge="cut", **KW
    )
    np.testing.assert_array_equal(out, expected)


def test_ties_go_to_the_patch_placed_first():
    # overlap 5 (odd): where two patches overlap, the middle column has equal weights
    image = coordinates(50, 50)
    out = predict_array(corner, image, patch_size=20, overlap=5, merge="cut", **KW)
    expected, windows = cut_reference(corner, image, 20, 5)
    np.testing.assert_array_equal(out, expected)
    assert [w[2] for w in windows[:3]] == [0, 15, 30]
    weights = gradient_weights(20, 5)
    # columns 15-19 are in the patches at left 0 and 15; at 17 both weigh 3/5 x row ramp
    assert weights[10, 17] == weights[10, 17 - 15]
    np.testing.assert_array_equal(out[1, :, 17], 1)  # the patch at left 0 (left + 1)
    np.testing.assert_array_equal(out[1, :, 18], 16)  # past the tie: the patch at left 15
    np.testing.assert_array_equal(out[1, :, 16], 1)
    # the same along rows
    np.testing.assert_array_equal(out[0, 17, :], 1)
    np.testing.assert_array_equal(out[0, 18, :], 16)


def test_clamped_last_patch_keeps_only_its_central_part():
    # windows at 0, 15, 30 and the clamped 33: the last one wins only where it is the most
    # central patch, not wherever it was placed last
    image = coordinates(53, 53)
    out = predict_array(corner, image, patch_size=20, overlap=5, merge="cut", **KW)
    expected, windows = cut_reference(corner, image, 20, 5)
    assert sorted({w[2] for w in windows}) == [0, 15, 30, 33]
    np.testing.assert_array_equal(out, expected)
    assert set(np.unique(out[1])) == {1, 16, 31, 34}
    np.testing.assert_array_equal(out[1, :, -1], 34)


@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("mask", [True, False])
def test_cut_with_nodata(shift, mask):
    image = scene(120, 150)
    image[:, :48, :48] = 0  # an all-nodata patch: skipped, its pixels covered by none or few
    expected, _ = cut_reference(offset, image, 32, 8, nodata=0, shift=shift, mask=mask)
    out = predict_array(
        offset,
        image,
        patch_size=32,
        overlap=8,
        nodata=0,
        shift_patches=shift,
        mask_nodata=mask,
        merge="cut",
        **KW,
    )
    np.testing.assert_array_equal(out, expected)
    if not shift:
        assert np.isnan(out[:, :16, :16]).all()  # no patch covered these


def test_cut_reduced_resolution_head():
    image = scene(130, 150)
    out = predict_array(
        Heads(), image, patch_size=32, overlap=8, nodata=0, merge="cut", **HEAD_KW, **KW
    )

    def quarter(x):
        return F.avg_pool2d(offset(x), 4)

    expected, _ = cut_reference(quarter, image, 32, 8, nodata=0, factor=4)
    assert out["quarter"].shape == (2, 33, 38)
    np.testing.assert_allclose(out["quarter"], expected, rtol=1e-6, atol=1e-6)
    full, _ = cut_reference(offset, image, 32, 8, nodata=0, multiple=4)  # windows on multiples of 4
    np.testing.assert_array_equal(out["full"], full)


def test_cut_per_output():
    image = scene(130, 150)
    run = dict(patch_size=32, overlap=8, nodata=0, **HEAD_KW, **KW)
    blend = predict_array(Heads(), image, **run)
    cut = predict_array(Heads(), image, merge="cut", **run)
    mixed = predict_array(Heads(), image, merge={"full": "cut"}, **run)
    np.testing.assert_array_equal(mixed["full"], cut["full"])
    np.testing.assert_array_equal(mixed["quarter"], blend["quarter"])
    np.testing.assert_array_equal(mixed["score"].values, blend["score"].values)
    assert not np.array_equal(cut["full"], blend["full"], equal_nan=True)
    # explicit "blend" is the default
    same = predict_array(Heads(), image, merge="blend", **run)
    np.testing.assert_array_equal(same["full"], blend["full"])


def test_cut_argmax_uses_the_cut_values():
    model = SeededConv()

    def patchwise(x):
        return model(x) + 3 * x.mean(dim=(1, 2, 3), keepdim=True) * torch.arange(4.0)[:, None, None]

    image = scene(100, 120)
    with torch.no_grad():
        probs = predict_array(
            patchwise, image, patch_size=32, overlap=12, nodata=0, merge="cut", **KW
        )
        classes = predict_array(
            patchwise,
            image,
            patch_size=32,
            overlap=12,
            nodata=0,
            merge="cut",
            output="argmax",
            **KW,
        )
    expected = np.nan_to_num(probs, nan=0.0).argmax(axis=0)[None].astype(np.uint8)
    valid = ~nodata_pixels(image, 0)
    np.testing.assert_array_equal(classes, expected * valid)


def test_cut_in_a_float16_mosaic():
    image = scene(97, 111, holes=False)
    out = predict_array(
        offset, image, patch_size=32, overlap=13, merge="cut", mosaic_dtype="float16", **KW
    )
    # weights compared as float16, values stored as float16
    expected, _ = cut_reference(offset, image, 32, 13, dtype=np.float16)
    np.testing.assert_array_equal(out, expected)


class Collect:
    def __init__(self):
        self.arrays = {}

    def write(self, name, top, block, shape, valid, factor=1):
        array = self.arrays.setdefault(name, np.zeros(shape, dtype=block.dtype))
        array[:, top : top + block.shape[1]] = block


def tiler(model, merge, patch_size=32, overlap=8, mosaic_dtype="float32", **outputs):
    runner = Runner(
        models=[model],
        device=torch.device("cpu"),
        dtype=torch.float32,
        normalizers={},
        nodata=0,
        primary="",
        make_specs=lambda names: make_specs(
            names, outputs.get("activation"), outputs.get("output")
        ),
    )
    return Tiler(
        runner,
        patch_size,
        overlap,
        2,
        torch.device("cpu"),
        True,
        False,
        mosaic_dtype=mosaic_dtype,
        merge=merge,
    )


@pytest.mark.parametrize("merge", ["cut", {"full": "cut"}, "blend"])
@pytest.mark.parametrize("mosaic_dtype", ["float32", "float16"])
def test_streaming_equals_in_memory(merge, mosaic_dtype):
    x = scene(310, 170)
    expected, _ = tiler(Heads(), merge, mosaic_dtype=mosaic_dtype, **HEAD_KW).predict(
        ArrayReader(x)
    )
    writer = Collect()
    tiler(Heads(), merge, mosaic_dtype=mosaic_dtype, **HEAD_KW).predict_blocks(
        ArrayReader(x), writer
    )
    for name in ("full", "quarter"):
        np.testing.assert_array_equal(writer.arrays[name], expected[name])


def test_streaming_argmax_cut():
    x = scene(250, 130)
    expected, _ = tiler(SeededConv(), "cut", output="argmax").predict(ArrayReader(x))
    writer = Collect()
    tiler(SeededConv(), "cut", output="argmax").predict_blocks(ArrayReader(x), writer)
    np.testing.assert_array_equal(writer.arrays[""], expected[""])


def test_predict_files_lazy_cut_writes_the_same_files(tmp_path):
    rio = pytest.importorskip("rasterio")
    from manypred.geo import RasterioReader

    x = (scene(150, 170) * 1000).astype(np.uint16)
    path = tmp_path / "scene.tif"
    profile = {"driver": "GTiff", "height": 150, "width": 170, "count": 3, "dtype": "uint16"}
    with rio.open(path, "w", **profile) as dst:
        dst.write(x)

    def load_array(p):
        with rio.open(p) as src:
            return src.read().astype(np.float32), src.profile

    def load_lazy(p):
        reader = RasterioReader(p)
        return reader, reader.profile

    kwargs = {"patch_size": 48, "overlap": 16, "nodata": 0, "merge": "cut", **HEAD_KW, **KW}
    predict_files(Heads(), [path], load_func=load_array, output_dir=tmp_path / "a", **kwargs)
    predict_files(Heads(), [path], load_func=load_lazy, output_dir=tmp_path / "b", **kwargs)
    reference = predict_array(Heads(), x.astype(np.float32), **kwargs)
    for name in ("full", "quarter"):
        with (
            rio.open(tmp_path / "a" / f"scene_pred_{name}.tif") as a,
            rio.open(tmp_path / "b" / f"scene_pred_{name}.tif") as b,
        ):
            np.testing.assert_array_equal(a.read(), b.read())
            np.testing.assert_array_equal(b.read(), reference[name])


def test_bad_merge_values():
    image = scene(64, 64)
    with pytest.raises(ValueError, match="merge must be one of"):
        predict_array(offset, image, patch_size=32, overlap=8, merge="mean", **KW)
    with pytest.raises(ValueError, match="merge must be one of"):
        predict_array(offset, image, patch_size=32, overlap=8, merge={"": "max"}, **KW)
    with pytest.raises(ValueError, match="unknown outputs"):
        predict_array(
            Heads(), image, patch_size=32, overlap=8, merge={"nope": "cut"}, **HEAD_KW, **KW
        )
    with pytest.raises(ValueError, match="image-level output 'score'"):
        predict_array(
            Heads(), image, patch_size=32, overlap=8, merge={"score": "cut"}, **HEAD_KW, **KW
        )
    with pytest.raises(ValueError, match="merge must be one of"):
        predict_files(offset, [], load_func=lambda p: p, merge="cutt", **KW)
    # a single-output model may name its head (or "") in the dict
    out = predict_array(offset, image, patch_size=32, overlap=8, merge={"": "cut"}, **KW)
    np.testing.assert_array_equal(
        out, predict_array(offset, image, patch_size=32, overlap=8, merge="cut", **KW)
    )
