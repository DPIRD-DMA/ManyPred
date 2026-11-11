"""Generate the golden files ManyPred's parity tests compare against.

Runs OmniCloudMask (OCM) at the pinned reference commit on small crops of OCM's test scenes and
stores inputs, parameters and outputs under ``tests/golden/`` (see ``tests/goldens.py`` for the
layout). Needs the ``parity`` dependency group and OCM's test scenes::

    uv sync --group test --group parity --extra geo
    uv run python scripts/make_goldens.py --scenes <dir>          # write tests/golden/
    uv run python scripts/make_goldens.py --scenes <dir> --check  # regenerate and compare

The OCM v4 weights are downloaded by OCM's own ``get_models`` into ``--weights``.
``--skip-ocm-models`` runs only the tiny-model cases.

All runs are on the CPU in float32 with 8 torch threads. Tiled cases use ``batch_size=1``,
because OCM's tiled path collects patches from a thread pool, so with larger batches the order
patches are accumulated in (and so the float rounding in overlaps) varies between runs; the one
case with ``batch_size=3`` is compared with a tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import warnings
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import goldens  # noqa: E402

OCM_COMMIT = "50b8f5f2e9d9446cd292df47548fb94c350db782"
DEFAULT_WEIGHTS = goldens.ocm_weights_dir()
TORCH_THREADS = 8

# windows in the source scenes (row offset, col offset, height, width)
S2_WINDOW = (6300, 0, 600, 760)  # OCM v4: 17% clear, 18% thick, 59% thin, 6% shadow
LC8_WINDOW = (2720, 560, 512, 640)  # scene edge, ~35% nodata, clear and cloudy

MODES = {
    "argmax": {"export_confidence": False},
    "probs": {"export_confidence": True, "softmax_output": True},
    "logits": {"export_confidence": True, "softmax_output": False},
}


def check_ocm_commit() -> None:
    """Refuse to run against any OCM but the pinned commit."""
    dist = importlib.metadata.distribution("omnicloudmask")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != OCM_COMMIT:
        raise SystemExit(
            f"omnicloudmask is installed from {commit or 'an unknown source'}, expected "
            f"{OCM_COMMIT}. Run `uv sync --group test --group parity --extra geo`."
        )


# --------------------------------------------------------------------------- inputs


def read_inputs(scenes: Path) -> dict[str, Any]:
    """Read the two base crops from OCM's test scenes (read only)."""
    import rasterio as rio
    from rasterio.windows import Window

    def read(paths: list[Path], window: tuple[int, int, int, int]):
        r, c, h, w = window
        bands = []
        for path in paths:
            with rio.open(path) as src:
                bands.append(src.read(1, window=Window(c, r, w, h)))
                geo = {
                    "crs": src.crs.to_wkt(),
                    "transform": list(src.window_transform(Window(c, r, w, h)))[:6],
                }
        return np.stack(bands), geo

    safe = next(scenes.glob("S2A_MSIL2A_*.SAFE"))
    s2_paths = [next(safe.rglob(f"*_{band}_10m.jp2")) for band in ("B04", "B03", "B08")]
    lc8_dir = scenes / "LC81960302014022LGN00"
    lc8_paths = [next(lc8_dir.rglob(f"*_{band}.TIF")) for band in ("B4", "B3", "B5")]

    s2, s2_geo = read(s2_paths, S2_WINDOW)
    lc8, lc8_geo = read(lc8_paths, LC8_WINDOW)
    assert s2.dtype == lc8.dtype == np.uint16
    return {"s2": s2, "lc8": lc8, "geo": {"s2": s2_geo, "lc8": lc8_geo}}


# --------------------------------------------------------------------------- cases


def image(source: str, rows: tuple[int, int], cols: tuple[int, int], dtype: str = "uint16"):
    return {"source": source, "rows": list(rows), "cols": list(cols), "dtype": dtype}


# Sizes are kept small so the float32 outputs fit the 20 MB budget; none is a multiple of
# the stride used on it
S2_FULL = image("s2", (0, 600), (0, 760))  # patch 512, overlap 150
S2_SUB = image("s2", (200, 360), (300, 500))  # 160 x 200, patch 64, overlap 20
S2_OCM = image("s2", (100, 292), (200, 456))  # 192 x 256, patch 128, overlap 32
S2_NARROW = image("s2", (400, 500), (600, 645))  # 100 x 45, narrower than the patch
LC8_FULL = image("lc8", (0, 512), (0, 640))  # all-nodata corner, diagonal edge
LC8_SUB = image("lc8", (0, 192), (160, 416))  # 192 x 256 across the nodata edge

CHIPS = {
    "size": 64,
    "chips": [
        ["s2", 0, 0],
        ["s2", 200, 300],
        ["lc8", 0, 0],  # entirely nodata
        ["s2", 400, 600],
        ["lc8", 100, 230],  # partly nodata
        ["s2", 500, 690],
        ["lc8", 300, 500],
    ],
}


def chips(dtype: str = "uint16") -> dict[str, Any]:
    return {**CHIPS, "dtype": dtype}


def build_cases(include_ocm: bool) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    def add(name, func, model, inputs, mode, **kwargs):
        if model == goldens.OCM_MODELS and not include_ocm:
            return
        cases[name] = {
            "func": func,
            "model": model,
            "input": inputs,
            "mode": mode,
            "kwargs": {**MODES[mode], **kwargs},
        }

    tiled = partial(add, func="predict_from_array")
    batch = partial(add, func="predict_from_batch")
    files = partial(add, func="predict_from_load_func")
    patch = {"patch_size": 64, "patch_overlap": 20, "batch_size": 1}
    lc8_patch = {"patch_size": 64, "patch_overlap": 16, "batch_size": 1}

    # tiled: a realistic patch size and OCM's overlap ratio
    for model in ("tiny0", goldens.OCM_MODELS):
        tiled(
            name=f"array_s2_large_argmax_{model}",
            model=model,
            inputs=S2_FULL,
            mode="argmax",
            patch_size=512,
            patch_overlap=150,
            batch_size=1,
        )
    tiled(
        name="array_lc8_argmax_large_tiny0",
        model="tiny0",
        inputs=LC8_FULL,
        mode="argmax",
        patch_size=128,
        patch_overlap=32,
        batch_size=1,
    )
    for mode in MODES:
        tiled(
            name=f"array_s2_overlap0_{mode}_tiny0",
            model="tiny0",
            inputs=S2_SUB,
            mode=mode,
            patch_size=64,
            patch_overlap=0,
            batch_size=1,
        )
        tiled(name=f"array_s2_{mode}_tiny0", model="tiny0", inputs=S2_SUB, mode=mode, **patch)
        tiled(
            name=f"array_s2_{mode}_tiny01", model="tiny0+tiny1", inputs=S2_SUB, mode=mode, **patch
        )
        tiled(name=f"array_lc8_{mode}_tiny0", model="tiny0", inputs=LC8_SUB, mode=mode, **lc8_patch)
    tiled(
        name="array_s2_probs_batch3_tiny0",
        model="tiny0",
        inputs=S2_SUB,
        mode="probs",
        **{**patch, "batch_size": 3},
    )
    for dtype in ("uint8", "float32"):
        tiled(
            name=f"array_s2_{dtype}_argmax_tiny0",
            model="tiny0",
            inputs={**S2_SUB, "dtype": dtype},
            mode="argmax",
            **patch,
        )
    tiled(
        name="array_s2_narrow_argmax_tiny0", model="tiny0", inputs=S2_NARROW, mode="argmax", **patch
    )
    tiled(
        name="array_lc8_nomask_logits_tiny0",
        model="tiny0",
        inputs=LC8_SUB,
        mode="logits",
        apply_no_data_mask=False,
        **lc8_patch,
    )
    # >30% nodata and a patch over half the scene: OCM shrinks the patch
    tiled(
        name="array_lc8_shrink_argmax_tiny0",
        model="tiny0",
        inputs=LC8_FULL,
        mode="argmax",
        patch_size=300,
        patch_overlap=100,
        batch_size=1,
    )
    for mode in ("argmax", "logits"):
        tiled(
            name=f"array_lc8_nan_{mode}_tiny0",
            model="tiny0",
            inputs={**LC8_SUB, "dtype": "float32_nan"},
            mode=mode,
            no_data_value=float("nan"),
            **lc8_patch,
        )
    tiled(
        name="array_s2_probs_ocm_v4",
        model=goldens.OCM_MODELS,
        inputs=S2_OCM,
        mode="probs",
        patch_size=128,
        patch_overlap=32,
        batch_size=1,
    )
    tiled(
        name="array_lc8_argmax_ocm_v4",
        model=goldens.OCM_MODELS,
        inputs=LC8_FULL,
        mode="argmax",
        patch_size=128,
        patch_overlap=32,
        batch_size=1,
    )
    tiled(
        name="array_lc8_probs_ocm_v4",
        model=goldens.OCM_MODELS,
        inputs=LC8_SUB,
        mode="probs",
        patch_size=128,
        patch_overlap=32,
        batch_size=1,
    )

    # batch: 7 chips in batches of 3, one entirely nodata and one partly
    for mode in MODES:
        batch(name=f"batch_{mode}_tiny0", model="tiny0", inputs=chips(), mode=mode, batch_size=3)
    for dtype in ("uint8", "float32"):
        batch(
            name=f"batch_{dtype}_argmax_tiny0",
            model="tiny0",
            inputs=chips(dtype),
            mode="argmax",
            batch_size=3,
        )
    for mode in ("argmax", "logits"):
        batch(
            name=f"batch_nan_{mode}_tiny0",
            model="tiny0",
            inputs=chips("float32_nan"),
            mode=mode,
            batch_size=3,
            no_data_value=float("nan"),
        )
    batch(
        name="batch_nomask_probs_tiny0",
        model="tiny0",
        inputs=chips(),
        mode="probs",
        batch_size=3,
        apply_no_data_mask=False,
    )
    batch(
        name="batch_probs_tiny01", model="tiny0+tiny1", inputs=chips(), mode="probs", batch_size=3
    )
    for mode in ("argmax", "probs"):
        batch(
            name=f"batch_{mode}_ocm_v4",
            model=goldens.OCM_MODELS,
            inputs=chips(),
            mode=mode,
            batch_size=3,
        )

    # load_func: two GeoTIFF scenes of different sizes, written and read back with OCM's
    # load_multiband
    scenes = {"scenes": [LC8_SUB, S2_SUB]}
    for mode in ("argmax", "probs"):
        files(name=f"files_{mode}_tiny0", model="tiny0", inputs=scenes, mode=mode, **lc8_patch)
    files(
        name="files_argmax_ocm_v4",
        model=goldens.OCM_MODELS,
        inputs=scenes,
        mode="argmax",
        **lc8_patch,
    )
    return cases


# --------------------------------------------------------------------------- running


def models_for(name: str, weights: Path) -> dict[str, Any]:
    """OCM keyword arguments selecting the case's model(s)."""
    if name == goldens.OCM_MODELS:
        # OCM's own download and loading path, v4 (latest at the pinned commit)
        return {"custom_models": None, "destination_model_dir": weights}
    seeds = [int(part.removeprefix("tiny")) for part in name.split("+")]
    return {"custom_models": [goldens.init_tiny(seed) for seed in seeds]}


def near_tie(logits: np.ndarray, class_axis: int) -> np.ndarray:
    """Flat indexes (over the argmax output's shape) whose top two logits are within
    NEAR_TIE."""
    top2 = np.sort(np.moveaxis(logits, class_axis, -1), axis=-1)[..., -2:]
    gap = top2[..., 1] - top2[..., 0]
    tie = np.expand_dims(gap < goldens.NEAR_TIE, class_axis)  # NaN gaps are not ties
    return np.flatnonzero(tie).astype(np.int64)


def write_tif(path: Path, array: np.ndarray, geo: dict[str, Any]) -> None:
    import rasterio as rio
    from affine import Affine

    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": geo["crs"],
        "transform": Affine(*geo["transform"]),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)


def profile_to_json(profile: Any) -> dict[str, Any]:
    out = {}
    for key, value in dict(profile).items():
        if key == "crs":
            value = value.to_wkt() if value is not None else None
        elif key == "transform":
            value = list(value)[:6]
        elif hasattr(value, "value"):  # enums
            value = value.value
        out[key] = value
    return out


def window_geo(geo: dict[str, Any], rows: list[int], cols: list[int]) -> dict[str, Any]:
    """Georeferencing of a sub-window of a base crop."""
    a, b, c, d, e, f = geo["transform"]
    return {
        "crs": geo["crs"],
        "transform": [a, b, c + a * cols[0] + b * rows[0], d, e, f + d * cols[0] + e * rows[0]],
    }


def run_case(case: dict[str, Any], inputs: dict[str, Any], weights: Path) -> dict[str, Any]:
    import omnicloudmask as ocm
    import rasterio as rio
    from omnicloudmask.data_loaders import load_multiband

    common = {"inference_device": "cpu", "inference_dtype": "float32", **case["kwargs"]}
    func = case["func"]
    out: dict[str, Any] = {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if func == "predict_from_array":
            data = goldens.make_image(inputs, case["input"])
            out["output"] = ocm.predict_from_array(
                data, **common, **models_for(case["model"], weights)
            )
        elif func == "predict_from_batch":
            data = goldens.make_chips(inputs, case["input"])
            out["output"] = ocm.predict_from_batch(
                data, **common, **models_for(case["model"], weights)
            )
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                paths = []
                for i, spec in enumerate(case["input"]["scenes"]):
                    path = tmp_path / f"scene_{i}.tif"
                    geo = window_geo(inputs["geo"][spec["source"]], spec["rows"], spec["cols"])
                    write_tif(path, goldens.make_image(inputs, spec), geo)
                    paths.append(path)
                outputs = ocm.predict_from_load_func(
                    scene_paths=paths,
                    load_func=partial(load_multiband, band_order=[1, 2, 3]),
                    output_dir=tmp_path / "out",
                    **common,
                    **models_for(case["model"], weights),
                )
                for i, path in enumerate(outputs):
                    with rio.open(path) as src:
                        out[f"output_{i}"] = src.read()
                        out[f"mask_{i}"] = src.read_masks(1)
                        out[f"profile_{i}"] = json.dumps(profile_to_json(src.profile))
                        out[f"tags_{i}"] = json.dumps(src.tags())
                out["output_names"] = json.dumps([p.name for p in outputs])

    case["warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})

    # where the class output could flip from float rounding alone
    if case["mode"] == "argmax":
        logits_case = {**case, "mode": "logits", "kwargs": {**case["kwargs"], **MODES["logits"]}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if func == "predict_from_load_func":
                for i, spec in enumerate(case["input"]["scenes"]):
                    single = {**logits_case, "func": "predict_from_array", "input": spec}
                    logits = run_case(single, inputs, weights)["output"]
                    out[f"near_tie_{i}"] = near_tie(logits, class_axis=0)
            else:
                logits = run_case(logits_case, inputs, weights)["output"]
                class_axis = 1 if func == "predict_from_batch" else 0
                out["near_tie"] = near_tie(logits, class_axis)
    return out


# --------------------------------------------------------------------------- writing


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out_dir: Path, scenes: Path, weights: Path, include_ocm: bool) -> dict[str, Any]:
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    inputs = read_inputs(scenes)
    nodata = [(chip == 0).all(axis=0).mean() for chip in goldens.make_chips(inputs, chips())]
    assert nodata[2] == 1 and 0 < nodata[4] < 1 and nodata.count(0) == 5, nodata
    np.savez_compressed(
        out_dir / "inputs.npz", s2=inputs["s2"], lc8=inputs["lc8"], geo=json.dumps(inputs["geo"])
    )
    weights_np = {
        f"{name}/{key}": value.numpy()
        for name in goldens.TINY_MODELS
        for key, value in goldens.init_tiny(int(name.removeprefix("tiny"))).state_dict().items()
    }
    np.savez_compressed(out_dir / "models.npz", **weights_np)

    summary = {}
    for name, case in build_cases(include_ocm).items():
        print(f"  {name}", flush=True)
        arrays = run_case(case, inputs, weights)
        np.savez_compressed(out_dir / "cases" / f"{name}.npz", meta=json.dumps(case), **arrays)
        summary[name] = {
            "func": case["func"],
            "model": case["model"],
            "mode": case["mode"],
            "warnings": case["warnings"],
            "deterministic": case["kwargs"].get("batch_size", 1) == 1
            or case["func"] == "predict_from_batch",
            "arrays": {
                key: [list(value.shape), str(value.dtype)]
                for key, value in arrays.items()
                if isinstance(value, np.ndarray) and value.dtype.kind != "U"
            },
        }

    files = sorted(p for p in out_dir.rglob("*.npz"))
    manifest = {
        "generated": date.today().isoformat(),
        "omnicloudmask_commit": OCM_COMMIT,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "segmentation_models_pytorch": importlib.metadata.version(
                "segmentation-models-pytorch"
            ),
            "timm": importlib.metadata.version("timm"),
            "rasterio": importlib.metadata.version("rasterio"),
        },
        "torch_threads": TORCH_THREADS,
        "device": "cpu",
        "dtype": "float32",
        "windows": {"s2": S2_WINDOW, "lc8": LC8_WINDOW},
        "near_tie": goldens.NEAR_TIE,
        "cases": summary,
        "sha256": {str(p.relative_to(out_dir)): sha256(p) for p in files},
        "total_bytes": sum(p.stat().st_size for p in files),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def compare(new_dir: Path, old_dir: Path) -> bool:
    """Compare regenerated goldens with the committed ones: exact where OCM is
    deterministic, else within float rounding."""
    old_manifest = json.loads((old_dir / "manifest.json").read_text())
    ok = True
    for rel in ("inputs.npz", "models.npz"):
        with np.load(new_dir / rel) as new, np.load(old_dir / rel) as old:
            for key in old.files:
                if not np.array_equal(new[key], old[key]):
                    print(f"DIFF {rel}:{key}")
                    ok = False
    for name, info in old_manifest["cases"].items():
        new_path = new_dir / "cases" / f"{name}.npz"
        if not new_path.exists():
            print(f"MISSING {name}")
            ok = False
            continue
        new, old = goldens_np(new_path), goldens_np(old_dir / "cases" / f"{name}.npz")
        for key, old_value in old.items():
            new_value = new.get(key)
            if new_value is None:
                print(f"MISSING {name}:{key}")
                ok = False
            elif info["deterministic"] or old_value.dtype.kind not in "f":
                same = np.array_equal(new_value, old_value, equal_nan=old_value.dtype.kind == "f")
                if not same:
                    print(f"DIFF {name}:{key}")
                    ok = False
            elif not np.allclose(new_value, old_value, atol=1e-5, equal_nan=True):
                print(f"DIFF {name}:{key} beyond 1e-5")
                ok = False
    print("goldens reproduce" if ok else "goldens DO NOT reproduce")
    return ok


def goldens_np(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", type=Path, required=True, help="OCM's tests/test data")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=goldens.GOLDEN_DIR)
    parser.add_argument("--check", action="store_true", help="regenerate and compare")
    parser.add_argument("--skip-ocm-models", action="store_true")
    args = parser.parse_args()

    check_ocm_commit()
    torch.set_num_threads(TORCH_THREADS)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            generate(Path(tmp), args.scenes, args.weights, not args.skip_ocm_models)
            sys.exit(0 if compare(Path(tmp), args.out) else 1)
    manifest = generate(args.out, args.scenes, args.weights, not args.skip_ocm_models)
    print(f"wrote {len(manifest['cases'])} cases, {manifest['total_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
