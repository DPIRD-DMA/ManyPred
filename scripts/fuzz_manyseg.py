"""Fuzz ManySeg -> save/load -> ManyPred: random ManySeg models, checked against invariants.

For each seed a random (but documented) ManySeg configuration is built: backbone, decoder,
single or several inputs (other resolutions, time series, metadata vectors with each
conditioning mode), fusion, heads (segmentation / regression / classification, reduced
resolution, uncertainty), padding. Every model is then checked:

1. ``roundtrip``: ``manyseg.save`` then ``manyseg.load`` gives bitwise-identical outputs;
2. ``spec``: ManyPred accepts the model's ``tiling_spec()``;
3. ``single_patch``: a scene that is one patch gives exactly what calling the model gives
   (every head: dense, reduced-resolution, image-level);
4. ``tiled``: an odd-sized scene with overlap runs; shapes match the spec (``ceil(H/f)`` for
   reduced heads), values are finite where the scene is valid, classes are uint8 (regression
   heads keep their values);
5. ``batch``: ``predict_batch`` equals calling the model, for two batch sizes;
6. ``lazy``: a lazy reader gives exactly the array result (image heads: every patch's values,
   window and valid fraction);
7. ``deterministic``: the same call twice gives the same result;
8. ``nodata``: with a nodata block (zeros, in every raster input), full-resolution heads are 0
   (or NaN where no patch covered) inside it and finite elsewhere, and a lazy reader again
   matches the array;
9. ``bfloat16`` (some seeds): inference in bfloat16 runs and gives finite values.

Configurations also vary what ManyPred is given: second rasters at half, the same or double
resolution (inferred, or passed as ``scales=``), and models pruned with ``select_heads`` before
saving. Not covered: fixed-size backbones, fp16, CUDA and MPS.

``create_model`` rejecting a configuration with a ValueError is recorded as ``rejected`` (not a
bug). Everything else that fails is a finding, written with its configuration and traceback.

    PYTHONPATH=<ManySeg>/packages/manyseg/src python scripts/fuzz_manyseg.py --seeds 0-49 \\
        --out fuzz.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import manyseg
import numpy as np
import torch

from manypred import normalize_torch, predict_array, predict_batch
from manypred.lazy import ArrayReader
from manypred.spec import validate_tiling_spec

BACKBONES = [
    "resnet18",
    "efficientnet_b0",
    "mobilenetv3_small_100",
    "pvt_v2_b0",
    "convnext_atto",
    "regnety_002",
]
DECODERS = ["unet", "fpn", "segformer", "unetplusplus", "deeplabv3plus", "upernet", "fastai_unet"]
FUSION_OPS = ["concat", "add", "gated", "attention", "diff", "concat_diff"]
COND_MODES = ["film", "head", "channels"]  # "token" is documented as not implemented
AGGREGATIONS = ["mean", "max", "attention"]
FIELDS = {
    "doy": {"type": "cyclic", "period": 365.25},
    "elev": {"type": "continuous", "mean": 40.0, "std": 10.0},
}
ATOL = 2e-4  # float rounding between differently batched convolutions


def sample(rng: random.Random) -> dict[str, Any]:
    """A random create_model configuration (plain data)."""
    cfg: dict[str, Any] = {
        "backbone": rng.choice(BACKBONES),
        "pretrained": False,
        "decoder": rng.choice(DECODERS),
    }
    if cfg["decoder"] == "unet" and rng.random() < 0.3:
        cfg["decoder_kwargs"] = {"out_stride": rng.choice([2, 4])}
    if rng.random() < 0.2:
        cfg["pad_to_stride"] = False
    layout = rng.choice(["single", "single", "two_rasters", "time_series", "metadata"])
    if layout == "single":
        cfg["in_channels"] = rng.randint(1, 6)
    elif layout == "two_rasters":
        second = {"in_channels": rng.randint(1, 4), "backbone": rng.choice(BACKBONES[:3])}
        if rng.random() < 0.5:
            second["alignment"] = "same_extent"
            # how the fuzzer draws it, and whether ManyPred is told (not create_model arguments)
            cfg["_scene_scale"] = rng.choice([0.5, 1.0, 2.0])
            cfg["_explicit_scales"] = rng.random() < 0.3
        cfg["inputs"] = {"a": {"in_channels": rng.randint(1, 5)}, "b": second}
        cfg["fusion"] = {"type": rng.choice(["intermediate", "late"]), "op": rng.choice(FUSION_OPS)}
    elif layout == "time_series":
        cfg["inputs"] = {
            "ts": {
                "kind": "timeseries",
                "in_channels": rng.randint(1, 4),
                "temporal": rng.choice(AGGREGATIONS),
            }
        }
    else:
        cfg["inputs"] = {
            "img": {"in_channels": rng.randint(1, 4)},
            "meta": {"kind": "vector", "fields": FIELDS, "conditioning": rng.choice(COND_MODES)},
        }
    if rng.random() < 0.3:
        cfg["num_classes"] = rng.randint(2, 5)
        return cfg
    heads: dict[str, Any] = {}
    for i in range(rng.randint(1, 3)):
        kind = rng.choice(["segmentation", "regression", "classification"])
        head: dict[str, Any] = {"type": kind, "out_channels": rng.randint(1, 4)}
        # the head type follows its scope: segmentation per pixel, classification per image,
        # regression either
        if kind == "classification" or (kind == "regression" and rng.random() < 0.3):
            head["scope"] = "image"
        elif rng.random() < 0.4:
            head["scale"] = rng.choice([0.5, 0.25])
        if kind == "regression" and rng.random() < 0.3:
            head["uncertainty"] = True
        heads[f"{kind[:3]}{i}"] = head
    cfg["heads"] = heads
    if len(heads) > 1 and rng.random() < 0.3:
        cfg["_select_heads"] = rng.sample(sorted(heads), rng.randint(1, len(heads) - 1))
    cfg["_bfloat16"] = rng.random() < 0.25
    return cfg


def make_scene(
    model: Any, cfg: dict[str, Any], height: int, width: int, rng: np.random.Generator
) -> dict[str, np.ndarray] | np.ndarray:
    """A random scene for the model's inputs (uint16 rasters, float vectors)."""
    if "inputs" not in cfg:
        return rng.integers(1, 10000, size=(cfg["in_channels"], height, width), dtype=np.uint16)
    spec = model.tiling_spec()["inputs"]
    scene: dict[str, np.ndarray] = {}
    for name, entry in cfg["inputs"].items():
        kind = entry.get("kind", "raster")
        if kind == "vector":
            scene[name] = (rng.random(len(entry["fields"])) * 100).astype(np.float32)
            continue
        scale = spec[name].get("scale") or cfg.get("_scene_scale", 0.5)  # same_extent
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        c = entry["in_channels"]
        if kind == "timeseries":
            scene[name] = rng.integers(1, 10000, size=(3, c, h, w), dtype=np.uint16)
        else:
            scene[name] = rng.integers(1, 10000, size=(c, h, w), dtype=np.uint16)
    return scene


# Inputs are z-scored per patch on both sides: raw 0-10000 values through untrained models give
# activations in the thousands, where batch-dependent kernels differ by whole units
NORM = {"type": "zscore", "scope": "tile"}


def normalize_spec(scene: Any) -> Any:
    if isinstance(scene, dict):
        return {k: NORM for k, v in scene.items() if v.ndim >= 3}
    return NORM


def as_batch(scene: Any) -> Any:
    """The model input for one scene, normalised as ManyPred does (time series per frame)."""

    def one(v: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(v.astype(np.float32))
        if v.ndim >= 3:  # a (T, C, H, W) series normalises each frame, as the batch
            return normalize_torch(t, NORM, nodata=None)[None]
        return t[None]

    if isinstance(scene, dict):
        return {k: one(v) for k, v in scene.items()}
    return one(scene)


def call_model(model: Any, scene: Any) -> dict[str, np.ndarray]:
    with torch.no_grad():
        out = model(as_batch(scene))
    out = out if isinstance(out, dict) else {"": out}
    return {k: v[0].numpy() for k, v in out.items()}


def as_dict(result: Any) -> dict[str, Any]:
    return result if isinstance(result, dict) else {"": result}


def values(result: Any) -> np.ndarray:
    """An output as an array (TilePredictions -> its per-patch values)."""
    return result.values[0] if hasattr(result, "values") else result


def same(a: Any, b: Any) -> bool:
    """Two outputs are identical (TilePredictions: values, windows and valid of every patch)."""
    if hasattr(a, "values"):
        return all(
            np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True)
            for k in ("values", "windows", "valid")
        )
    return np.array_equal(a, b, equal_nan=True)


def scales_for(cfg: dict[str, Any]) -> dict[str, Any]:
    """``scales=`` for predict_array when the configuration says to pass them."""
    if cfg.get("_explicit_scales"):
        return {"scales": {"b": cfg["_scene_scale"]}}
    return {}


def with_nodata(scene: Any, box: tuple[int, int, int, int], height: int) -> Any:
    """``scene`` with zeros (nodata) in ``box`` (top, bottom, left, right in primary pixels),
    scaled to each raster's own resolution."""

    def blank(v: np.ndarray) -> np.ndarray:
        if v.ndim < 3:
            return v
        v = v.copy()
        r = v.shape[-2] / height
        top, bottom, left, right = (round(x * r) for x in box)
        v[..., top:bottom, left:right] = 0
        return v

    if isinstance(scene, dict):
        return {k: blank(v) for k, v in scene.items()}
    return blank(scene)


RAW = {"activation": "none", "nodata": None, "device": "cpu", "progress": False}


def run_checks(seed: int, cfg: dict[str, Any], workdir: Path) -> list[dict[str, Any]]:
    findings = []

    def finding(check: str, message: str) -> None:
        findings.append({"seed": seed, "check": check, "config": cfg, "error": message})

    torch.manual_seed(seed)
    build = {k: v for k, v in cfg.items() if not k.startswith("_")}
    try:
        model = manyseg.create_model(**build).eval()
        if cfg.get("_select_heads"):
            model.select_heads(cfg["_select_heads"])
    except ValueError as e:
        return [{"seed": seed, "check": "rejected", "config": cfg, "error": str(e)}]
    except Exception:
        return [
            {"seed": seed, "check": "build_crash", "config": cfg, "error": traceback.format_exc()}
        ]
    rng = np.random.default_rng(seed)
    spec = model.tiling_spec()
    factors = [round(1 / o["scale"]) for o in spec["outputs"].values() if o.get("scale")]
    unit = int(np.lcm.reduce([spec["pad_multiple"], 2, *factors]))  # 2: half-res inputs
    extra = scales_for(cfg)
    # at least 32 px per unit: backbones downsample 32x even when the model doesn't pad
    unit *= -(-32 // unit)

    # 1. save / load
    try:
        acts = {n: "softmax" for n, h in model.heads.items() if h.kind != "regression"}
        path = manyseg.save(model, workdir / f"m{seed}.safetensors", activations=acts, normalize=[])
        loaded = manyseg.load(path)
        probe = make_scene(model, cfg, 2 * unit, 2 * unit, rng)
        a, b = call_model(model, probe), call_model(loaded, probe)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                finding("roundtrip", f"output {name!r} differs after save/load")
    except Exception:
        finding("roundtrip", traceback.format_exc())
        return findings

    # 2. the spec
    try:
        validate_tiling_spec(spec)
    except Exception:
        finding("spec", traceback.format_exc())
        return findings

    # 3. a scene that is one patch == calling the model
    size = 2 * unit
    scene = make_scene(model, cfg, size, size, rng)
    try:
        direct = call_model(model, scene)
        tiled = as_dict(
            predict_array(
                model,
                scene,
                patch_size=size,
                overlap=0,
                normalize=normalize_spec(scene),
                **extra,
                **RAW,
            )
        )
        for name, expected in direct.items():
            got = values(tiled[name])
            if got.shape != expected.shape:
                finding("single_patch", f"{name!r}: shape {got.shape} vs model {expected.shape}")
            elif not np.allclose(got, expected, atol=ATOL, rtol=ATOL):
                diff = float(np.abs(got - expected).max())
                finding("single_patch", f"{name!r}: max difference {diff:.3g}")
    except Exception:
        finding("single_patch", traceback.format_exc())

    # 4. an odd-sized scene, tiled with overlap
    height, width = 3 * unit + 7, 2 * unit + 13
    odd = make_scene(model, cfg, height, width, rng)
    patch, overlap = 2 * unit, unit  # windows on multiples of unit, half-patch overlap
    try:
        result = as_dict(
            predict_array(
                model,
                odd,
                patch_size=patch,
                overlap=overlap,
                batch_size=2,
                normalize=normalize_spec(odd),
                **extra,
                **RAW,
            )
        )
        for name, entry in spec["outputs"].items():
            key = name if name in result else ""
            out = result[key]
            if entry.get("scope") == "image":
                if not np.isfinite(out.values).all():
                    finding("tiled", f"{name!r}: non-finite image-level values")
                continue
            f = round(1 / entry.get("scale", 1.0))
            expected_hw = (-(-height // f), -(-width // f))
            if out.shape[1:] != expected_hw:
                finding("tiled", f"{name!r}: shape {out.shape} vs expected (*, {expected_hw})")
            elif not np.isfinite(out).all():
                finding("tiled", f"{name!r}: non-finite values on a scene without nodata")
        classes = as_dict(
            predict_array(
                model,
                odd,
                patch_size=patch,
                overlap=overlap,
                output="argmax",
                normalize=normalize_spec(odd),
                **extra,
                **{**RAW, "activation": None},
            )
        )
        for name, out in classes.items():
            if hasattr(out, "values"):
                continue
            # a global argmax leaves regression heads their values: their channels aren't classes
            regression = model.tiling_spec()["outputs"].get(name, {}).get("kind") == "regression"
            expected = np.float32 if regression else np.uint8
            if out.dtype != expected:
                finding("tiled", f"{name!r}: argmax gave {out.dtype}, not {expected.__name__}")
    except Exception:
        finding("tiled", traceback.format_exc())

    # 5. batch mode == calling the model
    try:
        chips = [make_scene(model, cfg, size, size, rng) for _ in range(3)]
        expected = [call_model(model, c) for c in chips]
        for batch_size in (1, 2):
            got = predict_batch(
                model, chips, batch_size=batch_size, normalize=normalize_spec(chips[0]), **RAW
            )
            got = as_dict(got)
            for name in expected[0]:
                stacked = np.stack([e[name] for e in expected])
                if got[name].shape != stacked.shape:
                    finding("batch", f"{name!r}: shape {got[name].shape} vs {stacked.shape}")
                elif not np.allclose(got[name], stacked, atol=ATOL, rtol=ATOL):
                    diff = float(np.abs(got[name] - stacked).max())
                    finding("batch", f"{name!r} batch_size={batch_size}: max diff {diff:.3g}")
    except Exception:
        finding("batch", traceback.format_exc())

    # 6. lazy readers == arrays, 7. determinism
    try:
        if isinstance(odd, dict):
            lazy_scene = {k: ArrayReader(v) if v.ndim >= 3 else v for k, v in odd.items()}
        else:
            lazy_scene = ArrayReader(odd)
        kwargs = dict(
            patch_size=patch,
            overlap=overlap,
            batch_size=2,
            normalize=normalize_spec(odd),
            **extra,
            **RAW,
        )
        first = as_dict(predict_array(model, odd, **kwargs))
        second = as_dict(predict_array(model, odd, **kwargs))
        lazy = as_dict(predict_array(model, lazy_scene, **kwargs))
        for name in first:
            if not same(first[name], second[name]):
                finding("deterministic", f"{name!r} differs between identical calls")
            if not same(first[name], lazy[name]):
                finding("lazy", f"{name!r} differs between a lazy reader and the array")
    except Exception:
        finding("lazy", traceback.format_exc())

    # 8. a nodata block: zeroed inside it (full-resolution heads), finite outside, lazy == array
    try:
        box = (0, patch + unit, 0, unit + 5)  # a whole patch plus a ragged edge
        holed = with_nodata(odd, box, height)
        if isinstance(holed, dict):
            lazy_holed = {k: ArrayReader(v) if v.ndim >= 3 else v for k, v in holed.items()}
        else:
            lazy_holed = ArrayReader(holed)
        kw = {**kwargs, "nodata": 0}
        eager = as_dict(predict_array(model, holed, **kw))
        lazy = as_dict(predict_array(model, lazy_holed, **kw))
        inside = np.zeros((height, width), bool)
        inside[box[0] : box[1], box[2] : box[3]] = True
        for name, entry in spec["outputs"].items():
            key = name if name in eager else ""
            if key not in eager:
                continue
            if not same(eager[key], lazy[key]):
                finding("nodata", f"{name!r}: a lazy reader differs from the array")
            if entry.get("scope") == "image" or entry.get("scale", 1.0) != 1.0:
                continue
            out = eager[key]
            block = out[:, inside]
            if not (np.isnan(block) | (block == 0)).all():  # zeroed, or NaN if never covered
                finding("nodata", f"{name!r}: nonzero values inside the nodata block")
            if not np.isfinite(out[:, ~inside]).all():
                finding("nodata", f"{name!r}: missing values outside the nodata block")
    except Exception:
        finding("nodata", traceback.format_exc())

    # 9. bfloat16 inference runs
    if cfg.get("_bfloat16"):
        try:
            low = as_dict(predict_array(model, odd, **{**kwargs, "dtype": "bfloat16"}))
            for name, out in low.items():
                v = out.values if hasattr(out, "values") else out
                if not np.isfinite(v).all():
                    finding("bfloat16", f"{name!r}: non-finite values")
            model.float()
        except Exception:
            finding("bfloat16", traceback.format_exc())
    return findings


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--out", type=Path, default=Path("fuzz.jsonl"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    # expected here: a global argmax beside image and regression heads, bfloat16 on the CPU
    warnings.filterwarnings("ignore", message="output='argmax' does not apply")
    warnings.filterwarnings("ignore", message="Inference on the CPU in torch.bfloat16")
    counts: dict[str, int] = {}
    import platform

    import manypred

    with tempfile.TemporaryDirectory() as tmp, args.out.open("w") as out:
        env = {
            "manypred": manypred.__version__,
            "manyseg": manyseg.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": args.threads,
            "platform": platform.platform(),
            "device": "cpu",
        }
        out.write(json.dumps({"check": "environment", **env}) + "\n")
        for seed in parse_seeds(args.seeds):
            cfg = sample(random.Random(seed))
            start = time.perf_counter()
            results = run_checks(seed, cfg, Path(tmp))
            for r in results:
                out.write(json.dumps(r) + "\n")
                counts[r["check"]] = counts.get(r["check"], 0) + 1
            status = ", ".join(r["check"] for r in results) or "ok"
            print(f"seed {seed:4d} {time.perf_counter() - start:5.1f}s  {status}", flush=True)
    print("summary:", json.dumps(counts))


if __name__ == "__main__":
    main()
