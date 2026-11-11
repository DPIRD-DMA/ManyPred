"""Throughput of OmniCloudMask's tiled path vs ManyPred's on the same scene.

Runs OCM's ``predict_from_array`` and ManyPred's ``predict_array`` (configured like OCM) with
OCM's v4 ensemble on a crop of an OCM test scene, alternating, after a warm-up, and prints the
median time per run. The GPU is shared, so the other processes' GPU use is printed with the
results and the numbers are "contended/exploratory".

    uv run python scripts/bench_tiled.py --scenes <dir> --size 4000 --batch-size 4 --dtype float16
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "scripts")]

import goldens  # noqa: E402
from ocm_compat import OCM_NORMALIZE  # noqa: E402

from manypred import predict_array  # noqa: E402


def gpu_state() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "n/a"


def read_scene(scenes: Path, size: int) -> np.ndarray:
    import rasterio as rio
    from rasterio.windows import Window

    safe = next(scenes.glob("S2A_MSIL2A_*.SAFE"))
    bands = []
    for band in ("B04", "B03", "B08"):
        with rio.open(next(safe.rglob(f"*_{band}_10m.jp2"))) as src:
            bands.append(src.read(1, window=Window(0, 4000, size, size)))
    return np.stack(bands)


def main() -> None:
    import omnicloudmask as ocm

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4000)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--scenes",
        type=Path,
        required=True,
        help="OCM's tests/test data",
    )
    args = parser.parse_args()

    image = read_scene(args.scenes, args.size)
    weights = goldens.ocm_weights_dir()
    models = goldens.load_ocm_models(weights)
    common = {"patch_size": args.patch_size, "batch_size": args.batch_size}

    def run_ocm():
        return ocm.predict_from_array(
            image,
            patch_overlap=args.overlap,
            inference_device=args.device,
            inference_dtype=args.dtype,
            destination_model_dir=weights,
            **common,
        )

    def run_manypred():
        return predict_array(
            models,
            image,
            overlap=args.overlap,
            device=args.device,
            dtype=args.dtype,
            normalize=OCM_NORMALIZE,
            output="argmax",
            progress=False,
            **common,
        )

    runs = {"ocm": run_ocm, "manypred": run_manypred}
    results = {name: run() for name, run in runs.items()}  # warm-up
    agreement = float((results["ocm"] == results["manypred"]).mean())
    times: dict[str, list[float]] = {name: [] for name in runs}
    gpu_before = gpu_state()
    for _ in range(args.repeats):
        for name, run in runs.items():
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            if args.device == "cuda":
                torch.cuda.synchronize()
            times[name].append(time.perf_counter() - start)
    gpu_after = gpu_state()

    megapixels = args.size * args.size / 1e6
    print(
        f"scene {args.size}x{args.size}, patch {args.patch_size}/{args.overlap}, "
        f"batch {args.batch_size}, {args.device} {args.dtype}, torch {torch.__version__}"
    )
    print(f"GPU (util, mem used, total) before: {gpu_before}; after: {gpu_after}")
    for name, values in times.items():
        median = statistics.median(values)
        print(
            f"{name:9s} median {median:6.2f} s  ({megapixels / median:5.1f} Mpx/s)  "
            f"runs {[round(v, 2) for v in values]}"
        )
    print(f"class agreement OCM vs ManyPred: {agreement:.6f}")


if __name__ == "__main__":
    main()
