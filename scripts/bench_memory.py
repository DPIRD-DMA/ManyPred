"""Peak RAM of predict_files on lazy scenes of growing size.

Each size runs in a fresh process: a synthetic reader computes each window on demand (no
scene in memory), a small conv net predicts it in patches, and the default GeoTIFF writer
writes finished rows as blocks. The child reports its peak resident set size (ru_maxrss).
For contrast, ``--eager`` loads the whole scene as an array instead.

    uv run python scripts/bench_memory.py --sizes 4096 16384 65536 --width 4096 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# large scenes are written here
OUT_DIR = (
    Path(os.environ.get("MANYPRED_BENCH_DIR", tempfile.gettempdir())) / "manypred_bench_memory"
)


class SyntheticReader:
    """A (3, H, W) uint16 scene computed from pixel coordinates, read window by window."""

    def __init__(self, height: int, width: int):
        self.shape = (3, height, width)
        self.dtype = np.dtype(np.uint16)

    def read(self, window):
        top, bottom, left, right = window
        rows = np.arange(top, bottom, dtype=np.float32)[:, None]
        cols = np.arange(left, right, dtype=np.float32)[None, :]
        base = 2000 + 1000 * np.sin(rows / 97.0) * np.cos(cols / 61.0)
        return np.stack([base, base * 1.1, base * 0.9]).astype(np.uint16)


def child(height: int, width: int, device: str, eager: bool) -> None:
    import torch

    from manypred import predict_files

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(16, 4, 1)
    )
    reader = SyntheticReader(height, width)

    def load(item):
        return (reader.read((0, height, 0, width)) if eager else reader), None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    [path] = predict_files(
        model,
        [OUT_DIR / f"scene_{height}x{width}.tif"],
        load_func=load,
        output_dir=OUT_DIR,
        patch_size=1024,
        overlap=256,
        batch_size=4,
        device=device,
        output="argmax",
        normalize={"type": "zscore", "scope": "tile"},
        progress=False,
    )
    seconds = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "height": height,
                "width": width,
                "eager": eager,
                "peak_mb": peak_mb,
                "seconds": seconds,
                "output_mb": Path(path).stat().st_size / 2**20,
            }
        )
    )
    Path(path).unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.width, args.device, args.eager)
        return
    for height in args.sizes:
        cmd = [
            sys.executable,
            __file__,
            "--child",
            str(height),
            "--width",
            str(args.width),
            "--device",
            args.device,
        ] + (["--eager"] if args.eager else [])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
