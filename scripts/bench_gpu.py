"""GPU throughput of ManyPred's tiled and batch paths, for optimisation.

For each configuration three timings are taken after a warm-up:

* ``manypred``: the full call (host scene -> patches -> device -> model -> blend -> host);
* ``model``: the model alone on the same number of same-size batches already on the device
  (a lower bound: what perfect overlap of everything else would give);
* ``pipeline``: the full call with a 1x1 conv instead of the model (ManyPred's own overhead).

Results are JSON lines on stdout. The model is ManySeg's "fast" preset (PVT-v2-B0 + SegFormer,
3.8 M parameters) with random weights, or a small conv net with ``--model conv``.

    python scripts/bench_gpu.py --suite baseline > results.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from manypred import predict_array, predict_batch
from manypred.windows import make_patch_indexes

NORMALIZE = {"type": "zscore", "scope": "tile"}


def build_model(name: str) -> torch.nn.Module:
    torch.manual_seed(0)
    if name == "fast":
        from manyseg import create_model

        return create_model(
            backbone="pvt_v2_b0",
            decoder="segformer",
            pretrained=False,
            in_channels=3,
            num_classes=4,
        ).eval()
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(32, 32, 3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(32, 4, 1),
    ).eval()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats: int) -> float:
    """Median seconds of ``repeats`` calls (after the caller's warm-up)."""
    times = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def model_only(model, batches: list[int], size: int, dtype: torch.dtype) -> float:
    """Seconds for the model alone on device-resident zeros, batch by batch."""
    model = model.to("cuda", dtype)
    inputs = {b: torch.zeros((b, 3, size, size), device="cuda", dtype=dtype) for b in set(batches)}
    with torch.no_grad():
        for b in set(batches):
            model(inputs[b])  # warm-up per shape (MIOpen kernel selection)

        def run():
            with torch.no_grad():
                for b in batches:
                    model(inputs[b])

        return timed(run, 2)


def tiled(args, model, scene, batch_size: int, dtype: str, **extra) -> dict:
    patches = len(set(make_patch_indexes(scene.shape[1], scene.shape[2], args.patch, args.overlap)))
    batches = [min(batch_size, patches - i) for i in range(0, patches, batch_size)]
    kwargs = dict(
        patch_size=args.patch,
        overlap=args.overlap,
        batch_size=batch_size,
        device="cuda",
        dtype=dtype,
        normalize=NORMALIZE,
        output="argmax",
        progress=False,
        nodata=0,
        **extra,
    )
    predict_array(model, scene, **kwargs)  # warm-up
    full = timed(lambda: predict_array(model, scene, **kwargs), args.repeats)
    conv = torch.nn.Conv2d(3, 4, 1)
    predict_array(conv, scene, **kwargs)
    pipeline = timed(lambda: predict_array(conv, scene, **kwargs), args.repeats)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    alone = model_only(model, batches, args.patch, torch_dtype)
    megapixels = scene.shape[1] * scene.shape[2] / 1e6
    return {
        "mode": "tiled",
        "batch_size": batch_size,
        "dtype": dtype,
        "patches": patches,
        "manypred_s": round(full, 3),
        "model_s": round(alone, 3),
        "pipeline_s": round(pipeline, 3),
        "mpx_per_s": round(megapixels / full, 1),
        "model_share": round(alone / full, 3),
        **{k: str(v) for k, v in extra.items()},
    }


def batch(args, model, chips, batch_size: int, dtype: str) -> dict:
    kwargs = dict(
        batch_size=batch_size,
        device="cuda",
        dtype=dtype,
        normalize=NORMALIZE,
        output="argmax",
        progress=False,
        nodata=0,
    )
    predict_batch(model, chips[: batch_size * 2], **kwargs)  # warm-up
    full = timed(lambda: predict_batch(model, chips, **kwargs), args.repeats)
    conv = torch.nn.Conv2d(3, 4, 1)
    predict_batch(conv, chips[: batch_size * 2], **kwargs)
    pipeline = timed(lambda: predict_batch(conv, chips, **kwargs), args.repeats)
    n = len(chips)
    batches = [min(batch_size, n - i) for i in range(0, n, batch_size)]
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    alone = model_only(model, batches, chips.shape[-1], torch_dtype)
    return {
        "mode": "batch",
        "batch_size": batch_size,
        "dtype": dtype,
        "chips": n,
        "manypred_s": round(full, 3),
        "model_s": round(alone, 3),
        "pipeline_s": round(pipeline, 3),
        "chips_per_s": round(n / full, 1),
        "model_share": round(alone / full, 3),
    }


def stages(args, model, scene, batch_size: int, dtype: str) -> dict:
    """Where the tiled time goes: planning (nodata map, windows), the patch loop, finishing
    (divide, argmax, mask, copy to the host). Times the Tiler's internals by wrapping them."""
    import manypred.tiled as tiled_module

    spent = {"nodata_map": 0.0, "plan": 0.0, "loop": 0.0, "finish": 0.0}

    def wrap(obj, name, key):
        original = getattr(obj, name)

        def timed_call(*a, **k):
            sync()
            start = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                sync()
                spent[key] += time.perf_counter() - start

        setattr(obj, name, timed_call)
        return original

    originals = [
        (tiled_module, "_nodata_map", wrap(tiled_module, "_nodata_map", "nodata_map")),
        (tiled_module, "plan_windows", wrap(tiled_module, "plan_windows", "plan")),
        (tiled_module.Runner, "run", wrap(tiled_module.Runner, "run", "loop")),
        (tiled_module, "finish", wrap(tiled_module, "finish", "finish")),
    ]
    kwargs = dict(
        patch_size=args.patch,
        overlap=args.overlap,
        batch_size=batch_size,
        device="cuda",
        dtype=dtype,
        normalize=NORMALIZE,
        output="argmax",
        progress=False,
        nodata=0,
    )
    try:
        predict_array(model, scene, **kwargs)  # warm-up
        for key in spent:
            spent[key] = 0.0
        total = timed(lambda: predict_array(model, scene, **kwargs), 1)
    finally:
        for obj, name, original in originals:
            setattr(obj, name, original)
    return {
        "mode": "stages",
        "batch_size": batch_size,
        "dtype": dtype,
        "total_s": round(total, 3),
        **{k: round(v, 3) for k, v in spent.items()},
        "other_s": round(total - sum(spent.values()), 3),
    }


def loop_breakdown(args, model, scene, batch_size: int, dtype: str) -> dict:
    """Where the patch loop's time goes, with a sync around each step (so the steps add up,
    but the total is a little higher than unsynchronised): waiting for the next prefetched
    batch, preparing it (move, uint16 restore, valid mask, normalise, cast), the model,
    and the sink (mosaic accumulation)."""
    import manypred.runner as runner_module

    spent = {"wait": 0.0, "prepare": 0.0, "model": 0.0, "sink": 0.0}
    original_prefetch = runner_module.prefetch
    original_ensemble = runner_module.ensemble
    original_predict = runner_module.Runner.predict

    def timed_prefetch(*a, **k):
        iterator = original_prefetch(*a, **k)
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                return
            spent["wait"] += time.perf_counter() - start
            yield item

    def timed_ensemble(models, x):
        sync()
        start = time.perf_counter()
        out = original_ensemble(models, x)
        sync()
        spent["model"] += time.perf_counter() - start
        return out

    def timed_predict(self, batch):
        sync()
        start = time.perf_counter()
        before = spent["model"]
        out = original_predict(self, batch)
        sync()
        spent["prepare"] += time.perf_counter() - start - (spent["model"] - before)
        return out

    def wrap_sink(sink):
        original_call = sink.__call__

        class Timed:
            to_host = sink.to_host

            def __getattr__(self, name):
                return getattr(sink, name)

            def __call__(self, prediction):
                sync()
                start = time.perf_counter()
                original_call(prediction)
                sync()
                spent["sink"] += time.perf_counter() - start

        return Timed()

    original_run = runner_module.Runner.run

    def timed_run(self, batches, sink):
        return original_run(self, batches, wrap_sink(sink))

    runner_module.prefetch = timed_prefetch
    runner_module.ensemble = timed_ensemble
    runner_module.Runner.predict = timed_predict
    runner_module.Runner.run = timed_run
    kwargs = dict(
        patch_size=args.patch,
        overlap=args.overlap,
        batch_size=batch_size,
        device="cuda",
        dtype=dtype,
        normalize=NORMALIZE,
        output="argmax",
        progress=False,
        nodata=0,
    )
    try:
        predict_array(model, scene, **kwargs)
        for key in spent:
            spent[key] = 0.0
        total = timed(lambda: predict_array(model, scene, **kwargs), 1)
    finally:
        runner_module.prefetch = original_prefetch
        runner_module.ensemble = original_ensemble
        runner_module.Runner.predict = original_predict
        runner_module.Runner.run = original_run
    return {
        "mode": "loop",
        "model": type(model).__name__,
        "batch_size": batch_size,
        "dtype": dtype,
        "total_s": round(total, 3),
        **{k: round(v, 3) for k, v in spent.items()},
    }


def model_variants(args, model, batch_size: int, dtype: str) -> dict:
    """The model alone on 196 batches-worth of patches: default vs channels_last."""
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype
    ]
    batches = [batch_size] * max(1, 196 // batch_size)
    default = model_only(model, batches, args.patch, torch_dtype)
    cl_model = build_model(args.model).to("cuda", torch_dtype).to(memory_format=torch.channels_last)
    x = torch.zeros((batch_size, 3, args.patch, args.patch), device="cuda", dtype=torch_dtype)
    x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        cl_model(x)

        def run():
            with torch.no_grad():
                for _ in batches:
                    cl_model(x)

        channels_last = timed(run, 2)
    return {
        "mode": "model",
        "batch_size": batch_size,
        "dtype": dtype,
        "find_mode": __import__("os").environ.get("MIOPEN_FIND_MODE", "default"),
        "default_s": round(default, 3),
        "channels_last_s": round(channels_last, 3),
        "patches": sum(batches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="baseline")
    parser.add_argument("--model", default="fast")
    parser.add_argument("--size", type=int, default=10980)
    parser.add_argument("--patch", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--chips", type=int, default=2048)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="torch.backends.cudnn.benchmark=True (MIOpen find with workspace)",
    )
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    print(
        json.dumps(
            {
                "env": {
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "hip": getattr(torch.version, "hip", None),
                    "python": platform.python_version(),
                    "model": args.model,
                    "scene": args.size,
                    "patch": args.patch,
                    "overlap": args.overlap,
                }
            }
        ),
        flush=True,
    )
    model = build_model(args.model)
    rng = np.random.default_rng(0)
    scene = rng.integers(1, 10000, size=(3, args.size, args.size), dtype=np.uint16)
    scene[:, : args.size // 10, : args.size // 10] = 0  # a nodata corner

    if args.suite in ("baseline", "tiled"):
        for dtype in args.dtypes:
            for batch_size in args.batch_sizes:
                print(json.dumps(tiled(args, model, scene, batch_size, dtype)), flush=True)
    if args.suite == "stages":
        for dtype in args.dtypes:
            for batch_size in args.batch_sizes:
                print(json.dumps(stages(args, model, scene, batch_size, dtype)), flush=True)
                print(json.dumps(model_variants(args, model, batch_size, dtype)), flush=True)
    if args.suite == "pipeline":
        conv = torch.nn.Conv2d(3, 4, 1)
        for dtype in args.dtypes:
            for batch_size in args.batch_sizes:
                print(json.dumps(stages(args, conv, scene, batch_size, dtype)), flush=True)
                print(json.dumps(loop_breakdown(args, conv, scene, batch_size, dtype)), flush=True)
                print(json.dumps(loop_breakdown(args, model, scene, batch_size, dtype)), flush=True)
    if args.suite in ("baseline", "batch"):
        chips = rng.integers(
            1, 10000, size=(args.chips, 3, args.chip_size, args.chip_size), dtype=np.uint16
        )
        for dtype in args.dtypes:
            for batch_size in (8, 32, 64):
                print(json.dumps(batch(args, model, chips, batch_size, dtype)), flush=True)


if __name__ == "__main__":
    main()
