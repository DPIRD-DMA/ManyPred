"""Dict outputs, scaled heads, image-level heads (TilePredictions) and postprocess."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from manypred import TilePredictions, predict_array, predict_batch

KW = {"device": "cpu", "progress": False, "nodata": None}


def coordinates(height, width):
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([rows + 1, cols + 1, np.full_like(rows, 7.0)])


def area_downsample(x, f):
    """Block mean over f x f blocks, ceil grid; partial edge blocks average zero padding (as a
    model padding its input to a multiple of f does)."""
    c, h, w = x.shape
    padded = np.zeros((c, -(-h // f) * f, -(-w // f) * f), dtype=np.float64)
    padded[:, :h, :w] = x
    return padded.reshape(c, padded.shape[1] // f, f, padded.shape[2] // f, f).mean(axis=(2, 4))


class MultiHead(torch.nn.Module):
    """seg: identity (B, 3, H, W); quarter: 1/4-scale block mean; score: image-level mean."""

    def forward(self, x):
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, -w % 4, 0, -h % 4))
        return {
            "seg": x,
            "quarter": F.avg_pool2d(padded, 4),
            "score": x.mean(dim=(2, 3)),
        }


# --- batch mode ------------------------------------------------------------------------------


def test_batch_dict_outputs():
    chips = np.stack([coordinates(32, 40), coordinates(32, 40) * 2])
    out = predict_batch(
        MultiHead(), chips, activation={"score": "sigmoid"}, output={"score": "probs"}, **KW
    )
    assert set(out) == {"seg", "quarter", "score"}
    np.testing.assert_allclose(out["seg"], chips)
    np.testing.assert_allclose(
        out["quarter"], np.stack([area_downsample(c, 4) for c in chips]), rtol=1e-5
    )
    np.testing.assert_allclose(out["score"], 1 / (1 + np.exp(-chips.mean(axis=(2, 3)))), rtol=1e-5)


def test_image_level_output_needs_explicit_activation():
    chips = coordinates(16, 16)[None]
    with pytest.raises(ValueError, match="image-level"):
        predict_batch(lambda x: x.mean(dim=(2, 3)), chips, **KW)
    raw = predict_batch(lambda x: x.mean(dim=(2, 3)), chips, output="logits", **KW)
    assert raw.shape == (1, 3)


def test_batch_postprocess_per_chip():
    chips = np.stack([coordinates(16, 16), coordinates(16, 16) + 1])
    out = predict_batch(torch.nn.Identity(), chips, postprocess=lambda p: float(p.sum()), **KW)
    assert out == pytest.approx([float(c.sum()) for c in chips])


# --- tiled mode: dense and scaled heads ------------------------------------------------------


@pytest.mark.parametrize(
    "height, width, patch, overlap",
    [
        (64, 64, 32, 0),
        (64, 80, 32, 8),
        (70, 90, 32, 8),
        (101, 77, 40, 12),
        (33, 131, 24, 16),
    ],
)
def test_scaled_head_tiled_equals_area_downsample(height, width, patch, overlap):
    image = coordinates(height, width)
    out = predict_array(
        MultiHead(),
        image,
        patch_size=patch,
        overlap=overlap,
        activation={"score": "sigmoid"},
        output={"score": "probs"},
        **KW,
    )
    np.testing.assert_allclose(out["seg"], image, rtol=1e-5)
    assert out["quarter"].shape == (3, -(-height // 4), -(-width // 4))
    expected = area_downsample(image, 4)
    # every block fully inside the scene is exact; partial edge blocks too, because edge
    # patches are padded with zeros like the model pads its input
    np.testing.assert_allclose(out["quarter"], expected, rtol=1e-5)


def test_scaled_patch_must_be_divisible():
    with pytest.raises(ValueError, match="divisible"):
        predict_array(
            MultiHead(),
            coordinates(64, 64),
            patch_size=30,
            overlap=8,
            activation={"score": "sigmoid"},
            output={"score": "probs"},
            **KW,
        )


def test_density_total_is_preserved():
    """A density head blended across seams keeps the analytical total (sum after blending)."""
    image = np.random.default_rng(0).random((1, 96, 120)).astype(np.float32)
    density = predict_array(torch.nn.Identity(), image, patch_size=32, overlap=12, **KW)
    assert density.sum() == pytest.approx(image.sum(), rel=1e-5)


def test_scaled_head_nodata_mask():
    image = coordinates(64, 64)
    image[:, :8] = 0  # two full rows of 4x4 blocks are nodata
    out = predict_array(
        MultiHead(),
        image,
        patch_size=32,
        overlap=8,
        nodata=0,
        device="cpu",
        progress=False,
        activation={"score": "sigmoid"},
        output={"score": "probs"},
    )
    assert np.isnan(out["quarter"][:, :2]).all()
    assert np.all(out["quarter"][:, 2:] > 0)


# --- tiled mode: image-level heads -----------------------------------------------------------


def score_model(x):
    return x.mean(dim=(2, 3))


def test_tile_predictions_structure():
    image = coordinates(64, 96)
    tp = predict_array(
        score_model, image, patch_size=32, overlap=0, activation="sigmoid", output="probs", **KW
    )
    assert isinstance(tp, TilePredictions)
    assert tp.values.shape == (6, 3) and tp.windows.shape == (6, 4) and tp.valid.shape == (6,)
    np.testing.assert_array_equal(tp.valid, 1.0)
    for value, (top, bottom, left, right) in zip(tp.values, tp.windows, strict=True):
        expected = 1 / (1 + np.exp(-image[:, top:bottom, left:right].mean(axis=(1, 2))))
        np.testing.assert_allclose(value, expected, rtol=1e-5)


def test_to_map_of_constant_head_is_constant():
    image = coordinates(70, 90)
    tp = predict_array(
        lambda x: torch.full((x.shape[0], 2), 0.25),
        image,
        patch_size=32,
        overlap=10,
        output="logits",
        **KW,
    )
    np.testing.assert_allclose(tp.to_map(), 0.25, rtol=1e-6)
    coarse = tp.to_map(stride=8)
    assert coarse.shape == (2, 9, 12)
    np.testing.assert_allclose(coarse, 0.25, rtol=1e-6)


def test_reduce_mean_is_valid_pixel_mean_of_the_map():
    image = coordinates(64, 64)
    image[:, :20] = 0
    tp = predict_array(
        score_model,
        image,
        patch_size=32,
        overlap=8,
        output="logits",
        device="cpu",
        progress=False,
        nodata=0,
    )
    full = tp.to_map()
    valid = ~(image == 0).all(axis=0)
    np.testing.assert_allclose(tp.reduce("mean"), full[:, valid].mean(axis=1), rtol=1e-5)


def test_reduce_max_ignores_all_nodata_and_sparse_tiles():
    image = coordinates(64, 64)
    image[:, :32, 32:] = 0  # top-right tile entirely nodata: never predicted
    image[:, 1:31, 1:31] = 0  # top-left tile nodata inside a valid ring: not shifted
    tp = predict_array(
        score_model,
        image,
        patch_size=32,
        overlap=0,
        output="logits",
        device="cpu",
        progress=False,
        nodata=0,
    )
    assert len(tp.values) == 3
    assert tp.valid.min() == pytest.approx(1 - 30 * 30 / 1024)
    keep = tp.valid >= 0.5
    np.testing.assert_allclose(tp.reduce("max", min_valid=0.5), tp.values[keep].max(axis=0))
    assert keep.sum() == 2
    with pytest.raises(ValueError, match="sum"):
        tp.reduce("sum")


def test_tiled_postprocess_runs_once_on_the_mosaic():
    calls = []

    def post(outputs):
        calls.append(outputs)
        return outputs["seg"].sum()

    image = coordinates(64, 64)
    total = predict_array(
        MultiHead(),
        image,
        patch_size=32,
        overlap=8,
        postprocess=post,
        activation={"score": "sigmoid"},
        output={"score": "probs"},
        **KW,
    )
    assert len(calls) == 1 and isinstance(calls[0]["score"], TilePredictions)
    assert total == pytest.approx(image.sum(), rel=1e-5)


def test_per_output_settings_reject_unknown_names():
    with pytest.raises(ValueError, match="unknown outputs"):
        predict_array(
            MultiHead(),
            coordinates(32, 32),
            patch_size=32,
            overlap=0,
            activation={"nope": "sigmoid"},
            **KW,
        )


# --- ManySeg end to end ----------------------------------------------------------------------


def test_manyseg_multi_head_model_runs_end_to_end():
    manyseg = pytest.importorskip("manyseg")
    torch.manual_seed(0)
    model = manyseg.create_model(
        backbone="resnet18",
        pretrained=False,
        in_channels=3,
        decoder="unet",
        heads={
            "seg": {"type": "segmentation", "out_channels": 3},
            "height": {"type": "regression", "out_channels": 1, "scale": 0.25},
            "scene": {"type": "classification", "out_channels": 5, "scope": "image"},
        },
    ).eval()
    image = np.random.default_rng(0).random((3, 150, 170)).astype(np.float32)
    kwargs = {
        "activation": {"seg": "softmax", "scene": "softmax"},
        "output": {"seg": "argmax", "scene": "probs"},
        **KW,
    }
    out = predict_array(model, image, patch_size=64, overlap=16, **kwargs)
    assert out["seg"].shape == (1, 150, 170) and out["seg"].dtype == np.uint8
    assert out["height"].shape == (1, 38, 43) and not np.isnan(out["height"]).any()
    assert isinstance(out["scene"], TilePredictions)
    np.testing.assert_allclose(out["scene"].reduce("mean").sum(), 1, rtol=1e-5)
    chips = np.stack([image[:, :64, :64], image[:, 64:128, 64:128]])
    batch = predict_batch(model, chips, **kwargs)
    assert batch["seg"].shape == (2, 1, 64, 64) and batch["height"].shape == (2, 1, 16, 16)
    assert batch["scene"].shape == (2, 5)


def test_batch_postprocess_runs_off_the_main_thread_for_generators():
    import threading

    threads = set()

    def post(chip):
        threads.add(threading.current_thread().name)
        return float(chip.sum())

    chips = [coordinates(16, 16), coordinates(16, 16) + 1]
    out = predict_batch(torch.nn.Identity(), (c for c in chips), postprocess=post, **KW)
    assert out == pytest.approx([float(c.sum()) for c in chips])
    assert threading.current_thread().name not in threads


def test_padded_edge_tiles_count_padding_as_invalid():
    """Windows snapped to multiples of 4 run past a 70-wide scene; their padding counts
    against the valid fraction."""

    def model(x):
        h, w = x.shape[-2:]
        return {"score": x.mean(dim=(2, 3)), "quarter": F.avg_pool2d(x, 4)}

    tp = predict_array(model, coordinates(64, 70), patch_size=32, overlap=0, output="logits", **KW)[
        "score"
    ]
    widths = tp.windows[:, 3] - tp.windows[:, 2]
    np.testing.assert_allclose(tp.valid, widths * 32 / 32**2)
    assert tp.valid.min() < 1


def test_padding_uses_zero_when_nodata_does_not_fit_the_dtype():
    from manypred.tiled import pad_value

    assert pad_value(np.zeros(1, np.uint8), -1) == 0
    assert pad_value(np.zeros(1, np.uint8), 0.5) == 0
    assert pad_value(np.zeros(1, np.uint8), float("nan")) == 0
    assert pad_value(np.zeros(1, np.uint16), 65535) == 65535
    assert np.isnan(pad_value(np.zeros(1, np.float32), float("nan")))
    assert pad_value(torch.zeros(1, dtype=torch.int16), -9999) == -9999


def test_no_extra_model_calls_without_reduced_outputs():
    """Models with only full-resolution outputs are never probed: one call per batch."""
    calls = []

    def model(x):
        calls.append(x.shape[0])
        if not x.any():
            raise AssertionError("called on a zero probe")
        return x

    image = coordinates(64, 96)
    predict_array(model, image, patch_size=32, overlap=0, batch_size=2, **KW)
    assert calls == [2, 2, 2]


def test_reduced_outputs_replan_once_then_are_cached():
    calls = []

    class Counting(MultiHead):
        def forward(self, x):
            calls.append(x.shape[0])
            return super().forward(x)

    from manypred import predict_files

    image = coordinates(64, 64)
    model = Counting()
    out = predict_files(
        model,
        [0, 1],
        load_func=lambda i: image,
        save_func=lambda p, *a: p,
        patch_size=32,
        overlap=8,
        activation={"score": "sigmoid"},
        output={"score": "probs"},
        device="cpu",
        progress=False,
        nodata=None,
    )
    np.testing.assert_allclose(out[1]["quarter"], area_downsample(image, 4), rtol=1e-5)
    # scene 1: first batch, then all batches after re-planning; scene 2: all batches
    batches = len(calls)
    assert batches == 1 + 9 + 9


def test_global_argmax_leaves_image_heads_as_values():
    """output='argmax' for a model with dense and image-level heads: classes for the dense
    head, per-patch values (with a warning) for the image head; asking argmax of the image
    head by name is an error."""
    image = coordinates(64, 64)

    def model(x):
        return {"seg": x[:, :2], "score": x.mean(dim=(2, 3))}

    with pytest.warns(UserWarning, match="does not apply to image-level output 'score'"):
        out = predict_array(
            model,
            image,
            patch_size=32,
            overlap=0,
            output="argmax",
            activation={"score": "sigmoid"},
            **KW,
        )
    assert out["seg"].dtype == np.uint8
    assert isinstance(out["score"], TilePredictions) and out["score"].values.shape == (4, 3)
    with pytest.raises(ValueError, match="per-patch values"):
        predict_array(
            model,
            image,
            patch_size=32,
            overlap=0,
            output={"seg": "argmax", "score": "argmax"},
            activation={"score": "sigmoid"},
            **KW,
        )


def test_auto_output_with_an_activation_gives_probabilities():
    """output="auto" with an activation passed (no tiling_spec) means "probs", not an error
    about output="logits", which the caller never asked for."""
    image = coordinates(32, 32)
    expected = torch.softmax(torch.from_numpy(image)[None], dim=1)[0].numpy()
    tiled = predict_array(
        torch.nn.Identity(), image, patch_size=16, overlap=4, activation="softmax", **KW
    )
    np.testing.assert_allclose(tiled, expected, rtol=1e-5)
    batch = predict_batch(torch.nn.Identity(), image[None], activation="softmax", **KW)
    np.testing.assert_allclose(batch[0], expected, rtol=1e-5)
    # an explicit output="logits" with an activation is still a contradiction
    with pytest.raises(ValueError, match="output='logits'"):
        predict_batch(torch.nn.Identity(), image[None], activation="softmax", output="logits", **KW)


def test_channels_last_images_are_refused_with_a_hint():
    model = torch.nn.Conv2d(3, 2, 1)
    with pytest.raises(ValueError, match="moveaxis"):
        predict_array(model, np.zeros((64, 64, 3), np.float32), patch_size=32, overlap=8, **KW)
    with pytest.raises(ValueError, match="moveaxis"):
        predict_batch(model, np.zeros((2, 64, 64, 3), np.float32), **KW)
    with pytest.raises(ValueError, match="moveaxis"):
        predict_batch(model, [np.zeros((64, 64, 3), np.float32)], **KW)


def test_dict_of_lists_chips_are_refused_clearly():
    chips = [coordinates(16, 16), coordinates(16, 16)]
    with pytest.raises(ValueError, match="list of"):
        predict_batch(lambda x: x["img"], {"img": chips}, **KW)


def test_image_level_warning_points_at_the_callers_line(tmp_path):
    """The global-output warning for image-level heads names the user's call, from
    predict_files too (it used a fixed stacklevel)."""
    from manypred import predict_files

    def model(x):
        return {"seg": x, "score": x.mean(dim=(2, 3))}

    image = coordinates(32, 32)
    kw = {**KW, "patch_size": 16, "overlap": 4, "output": "argmax"}
    with pytest.warns(UserWarning, match="image-level") as record:
        predict_files(model, ["a"], load_func=lambda _: image, save_func=lambda *a: None, **kw)
    assert {w.filename for w in record if "image-level" in str(w.message)} == {__file__}
    with pytest.warns(UserWarning, match="image-level") as record:
        predict_array(model, image, **kw)
    assert {w.filename for w in record if "image-level" in str(w.message)} == {__file__}
