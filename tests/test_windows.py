"""Window planning, patch reading and the blending sinks: edge cases found in review."""

import time
import warnings

import numpy as np
import pytest
import torch
from tqdm.auto import tqdm

from manypred import predict_array
from manypred import windows as windows_module
from manypred.lazy import PackedMask
from manypred.runner import set_up_run
from manypred.spec import output_factors
from manypred.tiled import AmbiguousFactor, BlockSink, Tiler, layout_of, pad_value, read_patch
from manypred.windows import make_patch_indexes, plan_windows, window_block

KW = {"device": "cpu", "progress": False}


def covered(windows, shape):
    out = np.zeros(shape, bool)
    for top, bottom, left, right in windows:
        out[top:bottom, left:right] = True
    return out


def packed(mask):
    return PackedMask(np.packbits(mask, axis=1), mask.shape[1])


@pytest.mark.parametrize("edge", ["bottom", "right", "top", "left"])
def test_shifted_windows_on_multiples_keep_every_valid_pixel(edge):
    """A window shifted up (or left) and then put back on a multiple of 4 used to round further
    away, dropping valid rows at its far edge that no other window covered."""
    nodata = np.zeros((64, 64), bool)
    nodata[
        {"bottom": np.s_[62:], "top": np.s_[:2], "right": np.s_[:, 62:], "left": np.s_[:, :2]}[edge]
    ] = True
    planned = plan_windows(nodata, 32, 0, multiple=4)
    assert all(t % 4 == 0 and lft % 4 == 0 for t, _, lft, _ in planned)
    assert not (~nodata & ~covered(planned, nodata.shape)).any()


def test_shifted_windows_on_multiples_predict_every_valid_pixel():
    image = np.ones((1, 64, 64), np.float32)
    image[:, 62:] = 0

    def model(x):  # an output at 1/4 resolution, so windows start at multiples of 4
        return torch.nn.functional.avg_pool2d(x, 4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = predict_array(model, image, patch_size=32, overlap=0, nodata=0, **KW)
    assert out.shape == (1, 16, 16)
    assert not np.isnan(out).any()  # row 15 holds image rows 60-61, which have data


@pytest.mark.parametrize("seed", range(5))
def test_planning_on_a_packed_mask_matches_the_array(seed):
    rng = np.random.default_rng(seed)
    nodata = np.zeros((301, 277), bool)
    nodata[: rng.integers(0, 80)] = True
    nodata[:, rng.integers(200, 277) :] = True
    nodata |= rng.random(nodata.shape) < 0.02
    for multiple in (1, 4):
        dense = plan_windows(nodata, 64, 16, multiple=multiple)
        assert plan_windows(packed(nodata), 64, 16, multiple=multiple) == dense
        assert not (~nodata & ~covered(dense, nodata.shape)).any()


def test_window_block_on_a_packed_mask_unpacks_only_the_window():
    rng = np.random.default_rng(0)
    nodata = rng.random((50, 203)) < 0.5
    mask = packed(nodata)
    for window in [(0, 50, 0, 203), (3, 17, 5, 13), (10, 60, 190, 230), (0, 1, 202, 203)]:
        top, bottom, left, right = window
        np.testing.assert_array_equal(window_block(mask, window), nodata[top:bottom, left:right])


def test_planning_a_lazy_scene_unpacks_little_more_than_the_scene(monkeypatch):
    """Planning used to unpack whole rows of the bit-packed map for every window and shift
    step: minutes for a 30000 x 30000 scene."""
    nodata = np.zeros((2000, 2000), bool)
    nodata[:, :5] = True  # every left-column window is shifted
    nodata[:3] = True
    unpacked = 0
    unpackbits = np.unpackbits

    def counting(*args, **kwargs):
        nonlocal unpacked
        out = unpackbits(*args, **kwargs)
        unpacked += out.size
        return out

    monkeypatch.setattr(windows_module.np, "unpackbits", counting)
    plan_windows(packed(nodata), 200, 50)
    windows = len(make_patch_indexes(2000, 2000, 200, 50))
    # each window, plus a patch around the shifted ones (a row and a column of windows)
    assert unpacked <= windows * 200 * 200 + 2 * (2000 // 150 + 1) * (3 * 200) ** 2


def test_block_sink_finds_final_rows_in_linear_time():
    """The rows each window finishes were a suffix minimum computed in quadratic time:
    half a minute for 20 000 windows."""
    windows = make_patch_indexes(30000, 30000, 128, 0)  # 55 225 windows
    windows = [windows[i] for i in np.random.default_rng(0).permutation(len(windows))]
    start = time.perf_counter()
    sink = BlockSink(
        windows=windows,
        nodata=None,
        writer=None,
        specs=lambda: None,
        mask=False,
        shape=(30000, 30000),
        patch_size=128,
        overlap=0,
        multiple=1,
        device=torch.device("cpu"),
        to_host=False,
        pbar=tqdm(disable=True),
    )
    assert time.perf_counter() - start < 2
    tops = [w[0] for w in windows]
    for i in (0, 1, 1000, len(windows) - 1):
        assert sink.final[i] == min(tops[i:])
    assert sink.final[-1] == 30000


def test_pad_value_of_a_boolean_scene():
    mask = np.zeros((1, 4, 4), bool)
    assert pad_value(mask, 0) is False and pad_value(mask, 1) is True
    assert pad_value(mask, 5) == 0 and pad_value(mask, float("nan")) == 0
    assert pad_value(torch.from_numpy(mask), 1) is True


def test_a_boolean_scene_with_nodata_tiles():
    image = np.random.default_rng(0).random((2, 50, 50)) > 0.5
    image[:, :, :10] = False

    def model(x):
        return x[:, :1] * 2

    out = predict_array(model, image, patch_size=32, overlap=8, nodata=0, **KW)
    assert out.shape == (1, 50, 50)
    assert np.isnan(out[:, :, :10]).all()


class NoDtypeReader:
    """A reader that doesn't say its dtype, returning integers."""

    def __init__(self, array):
        self.array, self.shape = array, array.shape

    def read(self, window):
        top, bottom, left, right = window
        return self.array[..., top:bottom, left:right]


def test_padding_follows_the_data_a_reader_returns():
    """Padding from a reader without a dtype assumed float32, so a NaN nodata failed to fill
    integer data."""
    reader = NoDtypeReader(np.ones((1, 10, 10), np.uint16))
    patch = read_patch(reader, (4, 12, 4, 12), float("nan"))
    assert patch.dtype == np.uint16 and patch.shape == (1, 8, 8)
    assert (patch[:, 6:] == 0).all() and (patch[:, :6, :6] == 1).all()


@pytest.mark.parametrize("name", ["patch_size", "overlap", "batch_size"])
@pytest.mark.parametrize("value", [32.0, True, "32"])
def test_sizes_must_be_integers(name, value):
    kwargs = {"patch_size": 32, "overlap": 8, "batch_size": 1, name: value}
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        predict_array(lambda x: x, np.zeros((1, 64, 64), np.float32), **kwargs, **KW)


@pytest.mark.parametrize("value", [2.0, True, "2"])
def test_batch_and_files_sizes_must_be_integers(value, tmp_path):
    """Only predict_array checked its sizes; elsewhere a float failed deep inside."""
    from manypred import predict_batch, predict_files

    chips = np.zeros((2, 1, 8, 8), np.float32)
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        predict_batch(lambda x: x, chips, batch_size=value, **KW)
    for name in ("batch_size", "patch_size", "overlap", "max_saves_in_flight"):
        with pytest.raises(TypeError, match=f"{name} must be an integer"):
            predict_files(
                lambda x: x,
                [tmp_path / "a.npy"],
                load_func=np.load,
                save_func=lambda *a: None,
                **{name: value},
                **KW,
            )


def test_numpy_integer_sizes_are_accepted():
    out = predict_array(
        lambda x: x, np.ones((1, 64, 64), np.float32), patch_size=np.int64(32), overlap=8, **KW
    )
    assert out.shape == (1, 64, 64)


def quarter(x):
    """A head at 1/4 resolution: ceil(P / 4) pixels for a P patch."""
    return torch.nn.functional.max_pool2d(x, 4, ceil_mode=True)  # padding is 0


def tiler_factors(model, image, **kwargs):
    """The Tiler's output factors for one scene (what the GeoTIFF writer is given), and its
    result."""
    setup = set_up_run(
        model,
        device="cpu",
        dtype="float32",
        batch_size=1,
        default_batch_size=1,
        spec=kwargs.pop("spec", None),
        normalize=None,
        activation=None,
        output="logits",
        nodata=None,
        compile=False,
    )
    tiler = Tiler(
        setup.runner,
        kwargs.pop("patch_size"),
        kwargs.pop("overlap"),
        1,
        torch.device("cpu"),
        True,
        progress=False,
        factors=tuple(output_factors(setup.tiling)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the patch shrinks to the scene
        result, _ = tiler.predict(image)
    return tiler.last_factors, result


@pytest.mark.parametrize("size", [5, 6, 7, 9])
def test_a_reduced_head_on_a_scene_smaller_than_the_patch_keeps_its_factor(size):
    """The patch shrinks to the scene, where the head's size fits several factors (a 2 x 2
    output of a 5 x 5 patch is 1/3 or 1/4): it was learnt as the smallest, 3."""
    image = np.ones((1, size, size), np.float32)
    factors, result = tiler_factors(quarter, image, patch_size=32, overlap=8)
    assert factors == {"": 4}
    expected = -(-size // 4)
    assert result[""].shape == (1, expected, expected)
    np.testing.assert_allclose(result[""], 1)


def test_a_declared_factor_settles_an_ambiguous_size():
    spec = {"outputs": {"": {"scope": "pixel", "scale": 0.25}}}
    # with the spec the full patch can't help (an 8 x 8 patch gives 2 x 2: 1/4 to 1/7)
    factors, _ = tiler_factors(
        quarter, np.ones((1, 5, 5), np.float32), patch_size=8, overlap=4, spec=spec
    )
    assert factors == {"": 4}


def test_a_full_patch_learns_the_factor_a_shrunk_one_cant():
    """On a 5 x 5 scene an 8 x 8 patch shrinks to 5 x 5, where the 2 x 2 output is 1/3 or
    1/4; the 8 x 8 patch it was asked for tells: 2 x 2 is exactly 1/4 of it."""
    factors, _ = tiler_factors(quarter, np.ones((1, 5, 5), np.float32), patch_size=8, overlap=4)
    assert factors == {"": 4}


def test_an_ambiguous_size_at_the_full_patch_is_an_error():
    # ceil(7 / f) = 2 for f from 4 to 6, and 7 is no multiple of any
    with pytest.raises(ValueError, match="fits a resolution of 1/4.*Declare its scale"):
        tiler_factors(quarter, np.ones((1, 20, 20), np.float32), patch_size=7, overlap=3)


def test_layout_of_takes_the_only_factor_that_fits():
    out = torch.zeros(1, 2, 250, 250)
    assert layout_of("a", out, 1000).factor == 4
    with pytest.raises(AmbiguousFactor):
        layout_of("a", torch.zeros(1, 2, 2, 2), 5)
    with pytest.raises(AmbiguousFactor):  # a shrunk patch needn't be a multiple of 3
        layout_of("a", torch.zeros(1, 2, 2, 2), 6, full=False)
    assert layout_of("a", torch.zeros(1, 2, 2, 2), 8).factor == 4  # the full patch: exact
    assert layout_of("a", torch.zeros(1, 2, 1, 1), 16).factor == 16
    assert layout_of("a", torch.zeros(1, 2, 2, 2), 5, declared=(4,)).factor == 4
    assert layout_of("a", torch.zeros(1, 2, 5, 5), 5).factor == 1


@pytest.mark.parametrize("nodata", [16777217, 16777217.0])
@pytest.mark.parametrize("kind", ["array", "tensor", "reader"])
def test_large_integer_nodata_is_compared_exactly(kind, nodata):
    """float32 can't tell 16777216 from 16777217; torch compared an int32 tensor with a
    float nodata in float32, so valid pixels were masked."""
    from manypred.lazy import ArrayReader, nodata_mask
    from manypred.lazy import nodata_fraction as reader_fraction
    from manypred.windows import nodata_fraction, nodata_pixels

    image = np.full((2, 3, 4), 16777216, np.int32)
    image[:, 0, 0] = 16777217
    expected = np.zeros((3, 4), bool)
    expected[0, 0] = True
    if kind == "reader":
        reader = ArrayReader(image)
        np.testing.assert_array_equal(nodata_mask(reader, nodata).unpacked(), expected)
        assert reader_fraction(reader, nodata) == pytest.approx(1 / 12)
        return
    x = torch.from_numpy(image) if kind == "tensor" else image
    np.testing.assert_array_equal(nodata_pixels(x, nodata), expected)
    assert nodata_fraction(x, nodata) == pytest.approx(1 / 12)
