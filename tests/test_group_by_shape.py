"""predict_batch(group_by_shape=True)."""

import numpy as np
import pytest
import torch
from helpers import SeededConv

from manypred import predict_batch

KW = {"device": "cpu", "progress": False, "normalize": {"type": "zscore", "scope": "tile"}}
RNG = np.random.default_rng(0)
SIZES = [(32, 32), (48, 40), (32, 32), (16, 64), (48, 40), (32, 32), (16, 64), (32, 32)]


def chips():
    return [RNG.random((3, h, w)).astype(np.float32) + 0.1 for h, w in SIZES]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_results_in_input_order_and_equal_to_single_calls(batch_size):
    data = chips()
    out = predict_batch(
        SeededConv(), data, batch_size=batch_size, group_by_shape=True, output="argmax", **KW
    )
    assert isinstance(out, list) and len(out) == len(data)
    for chip, pred in zip(data, out, strict=True):
        assert pred.shape == (1, *chip.shape[1:])
        expected = predict_batch(SeededConv(), chip[None], output="argmax", **KW)[0]
        np.testing.assert_array_equal(pred, expected)


def test_buckets_are_batched_by_shape():
    shapes = []

    class Recording(SeededConv):
        def forward(self, x):
            shapes.append(tuple(x.shape))
            return super().forward(x)

    predict_batch(Recording(), chips(), batch_size=2, group_by_shape=True, **KW)
    assert all(len({s[2:]}) == 1 for s in shapes)
    # 4 chips of 32 x 32, 2 of 48 x 40 and 2 of 16 x 64, in full batches of 2
    assert [s[0] for s in shapes] == [2, 2, 2, 2]


def test_same_shapes_still_give_a_list():
    """group_by_shape=True always returns one result per chip, whatever the shapes."""
    data = [RNG.random((3, 32, 32)).astype(np.float32) for _ in range(5)]
    grouped = predict_batch(SeededConv(), data, batch_size=2, group_by_shape=True, **KW)
    plain = predict_batch(SeededConv(), data, batch_size=2, **KW)
    assert isinstance(grouped, list) and len(grouped) == 5
    np.testing.assert_array_equal(np.stack(grouped), plain)
    stacked = predict_batch(SeededConv(), np.stack(data), batch_size=2, group_by_shape=True, **KW)
    assert isinstance(stacked, list)
    np.testing.assert_array_equal(np.stack(stacked), plain)


def test_with_load_func_and_dict_outputs():
    data = chips()

    def model(x):
        return {"seg": SeededConv()(x), "score": x.mean(dim=(2, 3))}

    out = predict_batch(
        model,
        range(len(data)),
        load_func=lambda i: data[i],
        batch_size=2,
        group_by_shape=True,
        output={"score": "logits"},
        **KW,
    )
    for chip, pred in zip(data, out, strict=True):
        assert pred["seg"].shape == (4, *chip.shape[1:])
        assert pred["score"].shape == (3,)


def test_postprocess_per_chip_in_input_order():
    data = chips()
    out = predict_batch(
        torch.nn.Identity(),
        data,
        batch_size=3,
        group_by_shape=True,
        postprocess=lambda p: p.shape,
        device="cpu",
        progress=False,
    )
    assert out == [(3, *c.shape[1:]) for c in data]


def test_without_grouping_mixed_shapes_suggest_it():
    with pytest.raises(ValueError, match="group_by_shape=True"):
        predict_batch(SeededConv(), chips(), **KW)


def test_many_distinct_shapes_do_not_pile_up():
    state = {"loaded": 0, "predicted": 0, "max_waiting": 0}
    shapes = [(16 + i, 16) for i in range(40)]  # every chip a different shape

    def load(i):
        state["loaded"] += 1
        return np.ones((3, *shapes[i]), np.float32)

    def model(x):
        state["predicted"] += x.shape[0]
        state["max_waiting"] = max(state["max_waiting"], state["loaded"] - state["predicted"])
        return x

    out = predict_batch(
        model,
        range(40),
        load_func=load,
        batch_size=2,
        group_by_shape=True,
        device="cpu",
        progress=False,
    )
    assert len(out) == 40
    # without a cap all 40 would wait; with it: 4 batches waiting, one loading, the prefetch
    assert state["max_waiting"] <= 4 * 2 + 2 + 2 * 2
