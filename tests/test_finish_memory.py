"""Finishing a mosaic: same results as the plain computation, without full-size temporaries."""

import numpy as np
import pytest
import torch
from helpers import SeededConv

from manypred import outputs, predict_array, tiled
from manypred.blend import Mosaic, gradient_weights
from manypred.windows import nodata_fraction

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
KW = {"patch_size": 64, "overlap": 16, "progress": False, "nodata": 0}


def image(size=300):
    rng = np.random.default_rng(0)
    x = rng.integers(1, 1000, size=(3, size, size + 20), dtype=np.uint16)
    x[:, :40, :60] = 0
    return x


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("chunk", [1, 37, 1000, 1 << 22])
def test_argmax_in_pieces_matches_the_whole(monkeypatch, device, dtype, chunk):
    monkeypatch.setattr(outputs, "FINISH_CHUNK", chunk)
    x = torch.randn(1, 4, 30, 50, generator=torch.Generator().manual_seed(0)).to(device, dtype)
    x[0, 2, 3, 4] = torch.nan
    x[0, 1, 5, 6] = torch.inf
    out = outputs.argmax(x, dim=1)
    assert out.dtype == torch.uint8 and out.shape == (1, 1, 30, 50)
    assert out.equal(torch.argmax(torch.nan_to_num(x, nan=0.0), dim=1, keepdim=True))


def test_argmax_of_more_than_256_classes_is_not_narrowed():
    x = torch.zeros(300, 2, 2)
    x[299] = 1
    assert outputs.argmax(x).dtype == torch.int64 and (outputs.argmax(x) == 299).all()


@pytest.mark.parametrize("size", [(0, 4, 2, 2), (2, 4, 0, 2)])
def test_argmax_and_its_check_take_empty_arrays(size):
    x = torch.zeros(size)
    outputs.check_argmax(x, torch.ones((size[0], 1, *size[2:]), dtype=torch.bool), "x")
    assert outputs.argmax(x, dim=1).shape == (size[0], 1, *size[2:])


@pytest.mark.parametrize("device", DEVICES)
def test_check_argmax_counts_only_nan_where_there_is_data(device):
    x = torch.zeros(1, 4, 2, 2, device=device)
    x[0, 1, 0, 0] = torch.nan
    data = torch.ones(1, 1, 2, 2, dtype=torch.bool)  # may stay on the host
    with pytest.raises(RuntimeError, match="NaN at 1 pixels"):
        outputs.check_argmax(x, data, "x")
    data[0, 0, 0, 0] = False
    outputs.check_argmax(x, data, "x")


def test_a_mosaic_without_weights_refuses_cut_and_keeps_the_same_sums():
    weights = gradient_weights(8, 2)
    with pytest.raises(ValueError, match="must track"):
        Mosaic(2, 16, 16, weights, torch.device("cpu"), merge="cut", track_weight=False)
    plain = Mosaic(2, 16, 16, weights, torch.device("cpu"))
    bare = Mosaic(2, 16, 16, weights, torch.device("cpu"), track_weight=False)
    x = torch.rand(2, 2, 8, 8, generator=torch.Generator().manual_seed(0))
    for mosaic in (plain, bare):
        mosaic.add(x, [(0, 8, 0, 8), (4, 12, 4, 12)])
    assert bare.weight is None and bare.sum.equal(plain.sum)


@pytest.mark.parametrize("merge", ["blend", "cut"])
def test_take_blended_gives_the_values_of_blended(merge):
    mosaic = Mosaic(2, 16, 16, gradient_weights(8, 2), torch.device("cpu"), merge=merge)
    x = torch.rand(1, 2, 8, 8, generator=torch.Generator().manual_seed(0))
    mosaic.add(x, [(0, 8, 0, 8)])  # the rest stays uncovered: NaN
    expected = mosaic.blended()
    assert np.array_equal(mosaic.take_blended().numpy(), expected.numpy(), equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("output", ["logits", "probs", "argmax"])
@pytest.mark.parametrize("chunk", [17, 1 << 22])
def test_small_pieces_do_not_change_results(monkeypatch, device, dtype, output, chunk):
    if device == "cpu" and dtype == "float16":
        pytest.skip("float16 inference is for accelerators")
    activation = {"activation": "softmax"} if output == "probs" else {}
    kw = {**KW, "device": device, "dtype": dtype, "output": output, **activation}
    expected = predict_array(SeededConv(), image(), **kw)
    monkeypatch.setattr(outputs, "FINISH_CHUNK", chunk)
    monkeypatch.setattr(tiled, "FINISH_CHUNK", chunk)
    out = predict_array(SeededConv(), image(), **kw)
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected, equal_nan=np.issubdtype(out.dtype, np.floating))


def test_argmax_mosaics_keep_no_weights():
    calls = []
    original = Mosaic.__init__

    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        calls.append(self.weight is None)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Mosaic, "__init__", spy)
        predict_array(SeededConv(), image(), output="argmax", device="cpu", **KW)
    assert calls == [True]


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("nodata", [0, float("nan"), None])
def test_nodata_fraction_matches_the_whole_array_rule(dtype, nodata):
    x = np.random.default_rng(0).integers(0, 4, size=(3, 40, 50)).astype(dtype)
    if dtype == np.float32:
        x[0, 1, 2], x[1, 3, 4] = np.nan, np.inf
    missing = ~np.isfinite(x) if np.issubdtype(dtype, np.floating) else np.zeros(x.shape, bool)
    if nodata == 0:
        missing |= x == 0
    assert nodata_fraction(x, nodata) == pytest.approx(missing.mean())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("output", ["argmax", "logits"])
def test_finishing_adds_little_to_the_gpu_peak(output):
    size = 2048
    x = np.random.default_rng(0).integers(1, 1000, size=(3, size, size), dtype=np.uint16)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    predict_array(
        SeededConv(),
        x,
        patch_size=256,
        overlap=64,
        batch_size=2,
        device="cuda",
        dtype="float16",
        mosaic_dtype="float16",
        output=output,
        normalize={"type": "zscore", "scope": "tile"},
        progress=False,
    )
    peak = torch.cuda.max_memory_allocated() - start
    sums = 4 * size * size * 2  # the float16 sums the mosaic must hold
    weights = size * size * 2
    # room for the model's batches and one piece, far below a float32 copy of the mosaic (67 MB)
    assert peak < sums + weights + 30e6, peak / 1e6
