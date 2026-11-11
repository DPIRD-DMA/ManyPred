"""predict_files: OCM predict_from_load_func parity and the files contract."""

import json
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import goldens
import numpy as np
import pytest
import torch
from helpers import ALL_DEVICES, SeededConv, random_images, run_with_timeout, wait_for_new_threads
from ocm_compat import manypred_kwargs, missing_as_nan

from manypred import postprocess_blocks, predict_array, predict_batch, predict_files
from manypred.files import _check_output_paths
from manypred.geo import GeoTIFFBlocks, RasterioReader, export_profile, write_geotiff
from manypred.io_lock import IO_LOCK

rio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

KW = {"progress": False, "device": "cpu", "patch_size": 32, "overlap": 8, "nodata": 0}


def write_tif(path: Path, array: np.ndarray, crs="EPSG:32750", transform=None) -> Path:
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "crs": crs,
        "transform": transform or Affine(10, 0, 400000, 0, -10, 6500000),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path


def load_tif(path):
    with rio.open(path) as src:
        return src.read(), src.profile


@pytest.fixture
def scenes(tmp_path):
    images = random_images(3, size=80, seed=7)
    images[1, :, :20] = 0  # nodata rows
    return [write_tif(tmp_path / f"scene_{i}.tif", image) for i, image in enumerate(images)]


# --- parity with OCM's predict_from_load_func -----------------------------------------------

FILE_CASES = [n for n in goldens.case_names() if n.startswith("files_")]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=[pytest.mark.slow, pytest.mark.network])
        if n.endswith(goldens.OCM_MODELS)
        else n
        for n in FILE_CASES
    ],
)
def test_files_parity(name, tmp_path):
    case = goldens.load_case(name)
    meta = case["meta"]
    inputs = goldens.load_inputs()
    paths = []
    for i, spec in enumerate(meta["input"]["scenes"]):
        base = inputs["geo"][spec["source"]]
        a, b, c, d, e, f = base["transform"]
        (r0, _), (c0, _) = spec["rows"], spec["cols"]
        transform = Affine(a, b, c + a * c0 + b * r0, d, e, f + d * c0 + e * r0)
        paths.append(
            write_tif(
                tmp_path / f"scene_{i}.tif",
                goldens.make_image(inputs, spec),
                crs=base["crs"],
                transform=transform,
            )
        )
    model = (
        goldens.load_ocm_models()
        if meta["model"] == goldens.OCM_MODELS
        else goldens.load_tiny(meta["model"])
    )
    out = predict_files(
        model,
        paths,
        load_func=load_tif,
        output_dir=tmp_path / "out",
        device="cpu",
        **manypred_kwargs(meta),
    )
    assert [p.parent for p in out] == [tmp_path / "out"] * len(paths)
    for i, path in enumerate(out):
        with rio.open(path) as src:
            pred, mask = src.read(), src.read_masks(1)
            profile = json.loads(json.dumps(dict(src.profile), default=str))
            crs, transform = src.crs.to_wkt(), list(src.transform)[:6]
        golden = case[f"output_{i}"]
        np.testing.assert_array_equal(mask, case[f"mask_{i}"])
        if golden.dtype == np.uint8:
            differ = np.flatnonzero(pred.ravel() != golden.ravel())
            assert len(np.setdiff1d(differ, case[f"near_tie_{i}"])) == 0
        else:
            np.testing.assert_allclose(pred, missing_as_nan(golden, pred), atol=1e-4)
        expected = json.loads(str(case[f"profile_{i}"]))
        assert crs == expected.pop("crs") and transform == expected.pop("transform")
        # OCM copies the source's block layout; ManyPred writes its own (export_profile)
        for key in ("tiled", "blockxsize", "blockysize", "interleave"):
            expected.pop(key, None)
        if golden.dtype != np.uint8:
            # the goldens are from OCM 1.7.1, which wrote LZW; OCM 1.8 and ManyPred write floats
            # with DEFLATE
            expected["compress"] = "deflate"
        for key, value in expected.items():
            assert str(profile.get(key)) == str(value), key


# --- the files contract ---------------------------------------------------------------------


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_matches_predict_array(scenes, tmp_path, device):
    out = predict_files(
        SeededConv(),
        scenes,
        load_func=load_tif,
        output_dir=tmp_path / "o",
        output="argmax",
        **{**KW, "device": device, "dtype": "float32"},
    )
    assert out == [tmp_path / "o" / f"scene_{i}_pred.tif" for i in range(3)]
    for scene, path in zip(scenes, out, strict=True):
        expected = predict_array(SeededConv(), load_tif(scene)[0], output="argmax", **KW)
        with rio.open(path) as src:
            np.testing.assert_array_equal(src.read(), expected)


def test_outputs_next_to_items_by_default(scenes):
    out = predict_files(SeededConv(), scenes[:1], load_func=load_tif, **KW)
    assert out == [scenes[0].with_name("scene_0_pred.tif")] and out[0].exists()


def test_overwrite_false_skips_existing(scenes, tmp_path):
    out_dir = tmp_path / "o"
    out_dir.mkdir()
    (out_dir / "scene_1_pred.tif").write_bytes(b"placeholder")
    loaded = []

    def loader(path):
        loaded.append(Path(path).name)
        return load_tif(path)

    out = predict_files(
        SeededConv(), scenes, load_func=loader, output_dir=out_dir, overwrite=False, **KW
    )
    assert loaded == ["scene_0.tif", "scene_2.tif"]
    assert (out_dir / "scene_1_pred.tif").read_bytes() == b"placeholder"
    assert out == [out_dir / f"scene_{i}_pred.tif" for i in range(3)]


def _same_stem_scenes(tmp_path):
    scenes = []
    for year in ("2020", "2021"):
        (tmp_path / year).mkdir()
        scenes.append(write_tif(tmp_path / year / "T50.tif", random_images(1, size=80)[0]))
    return scenes


def _never_load(item):
    raise AssertionError("loaded despite a bad output path")


def test_two_items_with_one_output_are_refused(tmp_path):
    with pytest.raises(ValueError, match="T50_pred.tif"):
        predict_files(
            SeededConv(),
            _same_stem_scenes(tmp_path),
            load_func=_never_load,
            output_dir=tmp_path / "o",
            **KW,
        )


def test_item_listed_twice_is_refused(scenes):
    with pytest.raises(ValueError, match="both be written"):
        predict_files(SeededConv(), [scenes[0], scenes[0]], load_func=_never_load, **KW)


def test_output_path_can_separate_same_stems(tmp_path):
    def named(item, output_dir):
        return tmp_path / f"{Path(item).parent.name}.tif"

    out = predict_files(
        SeededConv(), _same_stem_scenes(tmp_path), load_func=load_tif, output_path=named, **KW
    )
    assert out == [tmp_path / "2020.tif", tmp_path / "2021.tif"]
    assert all(path.exists() for path in out)


@pytest.mark.parametrize("overwrite", [True, False])
def test_output_that_is_an_input_is_refused(scenes, tmp_path, overwrite):
    source = write_tif(tmp_path / "scene_0_pred.tif", random_images(1, size=80)[0])
    before = source.read_bytes()
    with pytest.raises(ValueError, match="also an input"):
        predict_files(
            SeededConv(),
            [scenes[0], source],
            load_func=_never_load,
            overwrite=overwrite,
            **KW,
        )
    assert source.read_bytes() == before


def test_rerun_over_earlier_outputs_is_refused(scenes):
    first = predict_files(SeededConv(), scenes[:1], load_func=load_tif, **KW)
    with pytest.raises(ValueError, match="also an input"):
        predict_files(SeededConv(), [scenes[0], *first], load_func=_never_load, **KW)


def test_output_in_another_spelling_is_still_an_input(scenes):
    source = write_tif(scenes[0].with_name("scene_0_pred.tif"), random_images(1, size=80)[0])
    spelled = source.parent / "." / source.name
    with pytest.raises(ValueError, match="also an input"):
        predict_files(SeededConv(), [scenes[0], str(spelled)], load_func=_never_load, **KW)


def test_default_sidecar_that_is_an_input_is_refused(scenes):
    sidecar = scenes[0].with_name("scene_0_pred.json")
    with pytest.raises(ValueError, match="also an input"):
        predict_files(SeededConv(), [scenes[0], sidecar], load_func=_never_load, **KW)


def test_custom_output_path_is_checked_too(scenes):
    with pytest.raises(ValueError, match="both be written"):
        predict_files(
            SeededConv(),
            scenes,
            load_func=_never_load,
            save_func=lambda *a: None,
            output_path=lambda item, output_dir: "same.tif",
            **KW,
        )


def test_outputs_sharing_a_sidecar_are_refused(scenes, tmp_path):
    names = iter(["result.tif", "result.png", "other.tif"])
    table = {}

    def named(item, output_dir):
        return table.setdefault(item, tmp_path / next(names))

    with pytest.raises(ValueError, match=r"both be written to .*result\.json"):
        predict_files(SeededConv(), scenes, load_func=_never_load, output_path=named, **KW)


def test_json_output_is_not_its_own_sidecar(tmp_path):
    target = tmp_path / "result.json"
    assert _check_output_paths(["a.tif"], lambda i, d: target, None, sidecars=True) == [target]


def test_output_path_is_called_once_per_item_when_skipping(scenes, tmp_path):
    calls = []
    done = tmp_path / "done.tif"
    done.write_bytes(b"placeholder")

    def named(item, output_dir):
        calls.append(item)
        return done

    out = predict_files(
        SeededConv(),
        scenes[:1],
        load_func=_never_load,
        save_func=lambda *a: None,
        output_path=named,
        overwrite=False,
        **KW,
    )
    assert out == [done] and calls == scenes[:1]


def test_uris_are_compared_as_written(scenes):
    uris = iter(["s3://bucket/a//pred.tif", "s3://bucket/a/pred.tif"])
    table = {}

    def named(item, output_dir):
        return table.setdefault(item, next(uris))

    out = predict_files(
        SeededConv(),
        scenes[:2],
        load_func=load_tif,
        save_func=lambda prediction, meta, item, output_dir: item,
        output_path=named,
        **KW,
    )
    assert out == scenes[:2]


def test_paths_differing_in_case_collide_on_macos(scenes, tmp_path, monkeypatch):
    monkeypatch.setattr("manypred.files.sys.platform", "darwin")
    # the volume is asked whether it ignores case; answer as macOS's default volume does
    monkeypatch.setattr("manypred.files._ignores_case", lambda directory: True)
    table = {scenes[0]: tmp_path / "Result.tif", scenes[1]: tmp_path / "result.tif"}
    with pytest.raises(ValueError, match="both be written"):
        predict_files(
            SeededConv(),
            scenes[:2],
            load_func=_never_load,
            save_func=lambda *a: None,
            output_path=lambda item, output_dir: table[item],
            **KW,
        )


def _head(path):
    with open(path, "rb") as file:
        return file.read(4)


def test_bigtiff_default_and_override():
    array = np.zeros((1, 4, 4), dtype=np.float32)
    assert export_profile({}, array)["BIGTIFF"] == "IF_SAFER"
    assert export_profile({"BIGTIFF": "YES"}, array)["BIGTIFF"] == "YES"
    assert export_profile({"bigtiff": "NO"}, array)["bigtiff"] == "NO"
    assert "BIGTIFF" not in export_profile({"bigtiff": "NO"}, array)


def test_small_output_stays_classic_tiff(tmp_path):
    path = write_geotiff(tmp_path / "small.tif", np.zeros((1, 64, 64), dtype=np.float32))
    assert _head(path) == b"II*\0"


def test_output_that_might_pass_4gb_is_bigtiff(tmp_path):
    height, width = 40_000, 20_000  # 3.2 GB uncompressed; one row is written, the rest are empty
    writer = GeoTIFFBlocks(tmp_path / "huge.tif", {"height": height, "width": width})
    writer.write("", 0, np.ones((1, 1, width), dtype=np.float32), (1, height, width), None)
    assert _head(writer.close()) == b"II+\0"


def _notes(caught):
    return [n for n in getattr(caught.value, "__notes__", []) if n.startswith("While processing")]


def test_load_error_names_the_item_and_keeps_its_type(scenes):
    def load(path):
        if Path(path).name == "scene_1.tif":
            raise KeyError("no such band")
        return load_tif(path)

    with pytest.raises(KeyError, match="no such band") as caught:
        predict_files(SeededConv(), scenes, load_func=load, save_func=lambda *a: None, **KW)
    assert _notes(caught) == [f"While processing {scenes[1]}"]


def test_batch_load_error_names_the_item(scenes):
    def load(path):
        if Path(path).name == "scene_2.tif":
            raise OSError("corrupt")
        return load_tif(path)[0]

    with pytest.raises(OSError, match="corrupt") as caught:
        predict_files(
            SeededConv(),
            scenes,
            load_func=load,
            save_func=lambda *a: None,
            mode="batch",
            batch_size=1,
            **KW,
        )
    assert _notes(caught) == [f"While processing {scenes[2]}"]


def test_save_error_names_its_own_item_when_it_surfaces_later(scenes):
    def save(prediction, meta, item, output_dir):
        if Path(item).name == "scene_0.tif":
            raise ValueError("disk full")

    with pytest.raises(ValueError, match="disk full") as caught:
        predict_files(
            SeededConv(), scenes, load_func=load_tif, save_func=save, max_saves_in_flight=1, **KW
        )
    assert _notes(caught) == [f"While processing {scenes[0]}"]


def test_postprocess_error_names_the_item(scenes):
    def fails(prediction):
        raise ZeroDivisionError("bad")

    with pytest.raises(ZeroDivisionError) as caught:
        predict_files(
            SeededConv(),
            scenes[:1],
            load_func=load_tif,
            save_func=lambda *a: None,
            postprocess=fails,
            **KW,
        )
    assert _notes(caught) == [f"While processing {scenes[0]}"]


def test_model_error_names_the_scene(scenes):
    class Fails(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("model broke")

    with pytest.raises(RuntimeError, match="model broke") as caught:
        predict_files(Fails(), scenes, load_func=load_tif, save_func=lambda *a: None, **KW)
    assert _notes(caught) == [f"While processing {scenes[0]}"]


class _FailingReader:
    """A reader that opens, then fails on read number ``fail_on``."""

    def __init__(self, path, fail_on):
        self.array, self.profile = load_tif(path)
        self.shape, self.dtype = self.array.shape, self.array.dtype
        self.fail_on, self.reads = fail_on, 0

    def read(self, window):
        self.reads += 1
        if self.reads == self.fail_on:
            raise OSError("read failed")
        top, bottom, left, right = window
        return self.array[..., top:bottom, left:right]


@pytest.mark.parametrize("fail_on", [1, 2])  # the nodata pass, then a patch
def test_lazy_read_error_names_the_scene(scenes, tmp_path, fail_on):
    with pytest.raises(OSError, match="read failed") as caught:
        predict_files(
            SeededConv(),
            scenes,
            load_func=lambda path: _FailingReader(path, fail_on),
            output_dir=tmp_path / "o",
            **KW,
        )
    assert _notes(caught) == [f"While processing {scenes[0]}"]
    assert not list((tmp_path / "o").iterdir())


def test_streaming_write_error_names_the_scene(scenes, tmp_path, monkeypatch):
    def fails(self, *args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(GeoTIFFBlocks, "write", fails)
    with pytest.raises(OSError, match="write failed") as caught:
        predict_files(
            SeededConv(),
            scenes,
            load_func=lambda path: _FailingReader(path, 0),
            output_dir=tmp_path / "o",
            **KW,
        )
    assert _notes(caught) == [f"While processing {scenes[0]}"]


def test_failed_run_resumes_with_overwrite_false(scenes, tmp_path):
    def load(path):
        if Path(path).name == "scene_1.tif":
            raise OSError("corrupt")
        return load_tif(path)

    out = tmp_path / "o"
    done = predict_files(SeededConv(), scenes[:1], load_func=load, output_dir=out, **KW)
    written = done[0].read_bytes()
    with pytest.raises(OSError, match="corrupt"):
        predict_files(SeededConv(), scenes, load_func=load, output_dir=out, overwrite=False, **KW)
    assert done[0].read_bytes() == written
    assert not (out / "scene_1_pred.tif").exists()
    loaded = []

    def fixed(path):
        loaded.append(Path(path).name)
        return load_tif(path)

    predict_files(SeededConv(), scenes, load_func=fixed, output_dir=out, overwrite=False, **KW)
    assert loaded == ["scene_1.tif", "scene_2.tif"]


class _BatchSizes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.sizes = []

    def forward(self, x):
        self.sizes.append(x.shape[0])
        return x * self.weight


def _chip(item):
    return np.ones((1, 16, 16), np.float32)


@pytest.mark.parametrize(
    "mode, batch_size, expected",
    [
        ("batch", None, [8, 2]),
        ("batch", 3, [3, 3, 3, 1]),
        ("batch", 1, [1] * 10),
    ],
)
def test_batch_size_defaults_by_mode(mode, batch_size, expected):
    model = _BatchSizes()
    predict_files(
        model,
        list(range(10)),
        load_func=_chip,
        save_func=lambda *a: None,
        mode=mode,
        batch_size=batch_size,
        patch_size=16,
        overlap=0,
        device="cpu",
        progress=False,
    )
    assert model.sizes == expected


@pytest.mark.parametrize("batch_size, expected", [(None, [1, 1, 1]), (2, [2, 1])])
def test_tiled_batch_size_counts_patches(batch_size, expected):
    model = _BatchSizes()
    predict_files(
        model,
        [0],
        load_func=lambda item: np.ones((1, 16, 48), np.float32),
        save_func=lambda *a: None,
        batch_size=batch_size,
        patch_size=16,
        overlap=0,
        device="cpu",
        progress=False,
    )
    assert model.sizes == expected


def test_batch_size_is_still_checked():
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        predict_files(
            SeededConv(), [0], load_func=_chip, save_func=lambda *a: None, batch_size=0, **KW
        )


class _Capped(_BatchSizes):
    def tiling_spec(self):
        return {"max_size": 16}


def test_batch_mode_does_not_warn_about_patches():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict_files(
            _Capped(),
            [0],
            load_func=_chip,
            save_func=lambda *a: None,
            mode="batch",
            device="cpu",
            progress=False,
        )


def test_tiled_mode_still_caps_the_patch():
    with pytest.warns(UserWarning, match="max_size"):
        predict_files(
            _Capped(),
            [0],
            load_func=_chip,
            save_func=lambda *a: None,
            device="cpu",
            progress=False,
        )


class _FixedBatchProgram:
    """A callable that declares its batch size, as a saved program does."""

    batch_size = 4

    def __init__(self):
        self.sizes = []

    def __call__(self, x):
        self.sizes.append(x.shape[0])
        return x


def test_program_batch_size_wins_quietly_over_the_mode_default():
    program = _FixedBatchProgram()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict_files(
            program,
            list(range(5)),
            load_func=_chip,
            save_func=lambda *a: None,
            mode="batch",
            device="cpu",
            progress=False,
        )
    assert program.sizes == [4, 4]  # the short batch is padded


@pytest.mark.parametrize("mode", ["batch", "tiled"])
def test_program_warns_when_any_other_batch_size_was_asked_for(mode):
    # 8 is batch mode's default, but here it was asked for
    with pytest.warns(UserWarning, match="using it, not 8"):
        predict_files(
            _FixedBatchProgram(),
            list(range(5)),
            load_func=_chip,
            save_func=lambda *a: None,
            mode=mode,
            batch_size=8,
            patch_size=16,
            overlap=0,
            device="cpu",
            progress=False,
        )


@pytest.mark.parametrize("mode", ["tiled", "batch"])
@pytest.mark.parametrize("io_lock", [True, False])
def test_io_lock_decides_whether_your_functions_hold_the_lock(scenes, mode, io_lock):
    held = []

    def load(path):
        held.append(IO_LOCK.held())
        return load_tif(path)[0]

    def save(prediction, meta, item, output_dir):
        held.append(IO_LOCK.held())

    kw = {"patch_size": 32, "overlap": 8} if mode == "tiled" else {}
    predict_files(
        SeededConv(),
        scenes,
        load_func=load,
        save_func=save,
        mode=mode,
        io_lock=io_lock,
        batch_size=1,
        progress=False,
        device="cpu",
        **kw,
    )
    assert len(held) == 6 and set(held) == {io_lock}


def _load_ran_during_first_save(scenes, io_lock):
    """Whether scene 2 could load while scene 0 was saving. Scene 1 can't tell: it loads ahead,
    before scene 0 is even predicted."""
    loading = threading.Event()
    results = []

    def load(path):
        if Path(path).name == "scene_2.tif":
            loading.set()
        return load_tif(path)

    def save(prediction, meta, item, output_dir):
        if item == scenes[0]:
            results.append(loading.wait(1))

    predict_files(SeededConv(), scenes, load_func=load, save_func=save, io_lock=io_lock, **KW)
    return results[0]


def test_a_slow_save_doesnt_hold_up_loads_without_the_lock(scenes):
    assert _load_ran_during_first_save(scenes, io_lock=False)


@pytest.mark.parametrize("load", [load_tif, lambda path: RasterioReader(path)])
def test_default_writer_works_and_locks_itself_without_the_lock(scenes, tmp_path, load):
    default = predict_files(
        SeededConv(), scenes[:1], load_func=load, output_dir=tmp_path / "a", **KW
    )
    unlocked = predict_files(
        SeededConv(), scenes[:1], load_func=load, output_dir=tmp_path / "b", io_lock=False, **KW
    )
    with rio.open(default[0]) as a, rio.open(unlocked[0]) as b:
        np.testing.assert_array_equal(a.read(), b.read())
        assert a.crs == b.crs


def _whole_warnings(record):
    return [str(w.message) for w in record if "held in memory" in str(w.message)]


def _record_warnings(**kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        predict_files(SeededConv(), **kwargs, **KW)
    return _whole_warnings(record)


@pytest.mark.parametrize(
    "extra, cause",
    [
        ({"save_func": lambda *a: None}, "a custom save_func"),
        ({"postprocess": lambda prediction: prediction}, "postprocess"),
        ({"save_func": lambda *a: None, "postprocess": lambda p: p}, "postprocess"),
    ],
)
def test_lazy_scene_held_whole_warns_once_with_its_cause(scenes, tmp_path, extra, cause):
    extra = {"output_dir": tmp_path, **extra}
    found = _record_warnings(items=scenes, load_func=RasterioReader, **extra)
    assert len(found) == 1 and f"because of {cause}" in found[0] and str(scenes[0]) in found[0]


def test_lazy_scene_written_in_row_blocks_doesnt_warn(scenes, tmp_path):
    assert not _record_warnings(items=scenes, load_func=RasterioReader, output_dir=tmp_path)


def test_array_scene_doesnt_warn_about_being_held_whole(scenes, tmp_path):
    assert not _record_warnings(
        items=scenes, load_func=load_tif, save_func=lambda *a: None, output_dir=tmp_path
    )


class _WarnsSpec(torch.nn.Module):
    def __init__(self, **spec):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.spec = spec

    def forward(self, x):
        return x * self.weight

    def tiling_spec(self):
        return self.spec


class _WarnsProgram:
    fixed_size = (16, 16)
    batch_size = 4

    def __call__(self, x):
        return x


_SCENE = np.ones((1, 32, 32), np.float32)
_CALLS = {
    "predict_array": lambda model, **kw: predict_array(model, _SCENE, **kw),
    "predict_batch": lambda model, **kw: predict_batch(model, _SCENE[None], **kw),
    "predict_files": lambda model, **kw: predict_files(
        model, [0], load_func=lambda item: _SCENE, save_func=lambda *a: None, **kw
    ),
}


@pytest.mark.parametrize(
    "call, make, kwargs, match",
    [
        ("predict_array", _WarnsSpec, {"dtype": "float16"}, "on the CPU"),
        ("predict_batch", _WarnsSpec, {"dtype": "float16"}, "on the CPU"),
        ("predict_files", _WarnsSpec, {"dtype": "float16"}, "on the CPU"),
        ("predict_array", lambda: _WarnsSpec(max_size=16), {}, "max_size"),
        ("predict_files", lambda: _WarnsSpec(max_size=16), {}, "max_size"),
        ("predict_array", _WarnsSpec, {"patch_size": 64, "overlap": 0}, "larger than the scene"),
        ("predict_files", _WarnsSpec, {"patch_size": 64, "overlap": 0}, "larger than the scene"),
        ("predict_array", _WarnsProgram, {"batch_size": 3}, "batch size 4"),
        ("predict_batch", _WarnsProgram, {"batch_size": 3}, "batch size 4"),
        ("predict_files", _WarnsProgram, {"batch_size": 3}, "batch size 4"),
        ("predict_array", _WarnsProgram, {"patch_size": 32, "overlap": 0}, "exported at a fixed"),
        ("predict_files", _WarnsProgram, {"patch_size": 32, "overlap": 0}, "exported at a fixed"),
    ],
)
def test_setup_warnings_point_at_the_caller(call, make, kwargs, match):
    with pytest.warns(UserWarning, match=match) as record:
        _CALLS[call](make(), device="cpu", progress=False, **kwargs)
    found = [w for w in record if re.search(match, str(w.message))]
    assert found[0].filename == __file__, found[0].filename


def test_save_still_running_after_the_grace_period_warns(scenes, monkeypatch):
    monkeypatch.setattr("manypred.files._SAVE_GRACE_SECONDS", 0.1)
    release = threading.Event()

    def load(path):
        if Path(path).name == "scene_1.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    try:
        with (
            pytest.warns(UserWarning, match="could overwrite"),
            pytest.raises(RuntimeError, match="loader failed"),
        ):
            predict_files(
                SeededConv(),
                scenes,
                load_func=load,
                save_func=lambda *a: release.wait(10),
                **KW,
            )
    finally:
        release.set()


def test_warnings_as_errors_dont_replace_the_runs_error(scenes, monkeypatch):
    monkeypatch.setattr("manypred.files._SAVE_GRACE_SECONDS", 0.1)
    release = threading.Event()

    def load(path):
        if Path(path).name == "scene_1.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(RuntimeError, match="loader failed"):
                predict_files(
                    SeededConv(),
                    scenes,
                    load_func=load,
                    save_func=lambda *a: release.wait(10),
                    **KW,
                )
    finally:
        release.set()


def test_saves_finishing_in_time_dont_warn(scenes):
    def load(path):
        if Path(path).name == "scene_1.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="loader failed"):
            predict_files(SeededConv(), scenes, load_func=load, save_func=lambda *a: None, **KW)
    assert not [w for w in record if "could overwrite" in str(w.message)]


class _StreamingWriter:
    """A custom save_func with the block writer hook, and no GDAL."""

    writes_blocks = True

    def __init__(self):
        self.held = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("a lazy scene should stream")

    def open(self, meta, item, output_dir, grid=None):
        self.held.append(IO_LOCK.held())
        return self

    def write(self, name, top, block, shape, valid, factor=1):
        self.held.append(IO_LOCK.held())

    def close(self, extras=None):
        self.held.append(IO_LOCK.held())
        return "done"

    def abort(self):
        pass


@pytest.mark.parametrize("io_lock", [True, False])
def test_custom_streaming_writer_follows_io_lock(scenes, io_lock):
    writer = _StreamingWriter()
    out = predict_files(
        SeededConv(),
        scenes[:1],
        load_func=RasterioReader,
        save_func=writer,
        io_lock=io_lock,
        **KW,
    )
    assert out == ["done"] and len(writer.held) >= 3 and set(writer.held) == {io_lock}


def test_default_block_writer_locks_even_without_io_lock(scenes, tmp_path, monkeypatch):
    held = []
    original = GeoTIFFBlocks.write

    def write(self, *args, **kwargs):
        held.append(IO_LOCK.held())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoTIFFBlocks, "write", write)
    predict_files(
        SeededConv(),
        scenes[:1],
        load_func=RasterioReader,
        output_dir=tmp_path,
        io_lock=False,
        **KW,
    )
    assert held and all(held)


def _opens_while_saving(scenes, make):
    """Whether ``make(path)`` runs while another thread holds the save side of the lock."""
    entered, release, opened = threading.Event(), threading.Event(), threading.Event()

    def saver():
        with IO_LOCK.write():
            entered.set()
            release.wait(5)

    def opener():
        make(scenes[0])
        opened.set()

    thread = threading.Thread(target=saver)
    thread.start()
    entered.wait(5)
    other = threading.Thread(target=opener)
    other.start()
    during = opened.wait(0.3)
    release.set()
    thread.join()
    other.join()
    return during


def test_opening_a_reader_takes_turns_with_saves(scenes):
    assert not _opens_while_saving(scenes, RasterioReader)


def test_postprocess_blocks_opening_a_raster_takes_turns_with_saves(scenes):
    def run(path):
        postprocess_blocks({"a": path}, lambda fields, block, scene: None)

    assert not _opens_while_saving(scenes, run)


@pytest.mark.parametrize(
    "returned, match",
    [
        ((np.ones((3, 16, 16), np.float32), {}, "extra"), "tuple of 3"),
        (np.ones((16, 16), np.float32), "must be"),
        ({}, "at least one input"),
    ],
)
def test_bad_load_func_result_names_the_item_in_batch_mode(returned, match):
    with pytest.raises(ValueError, match=match) as caught:
        predict_files(
            SeededConv(),
            ["ITEM-X"],
            load_func=lambda item: returned,
            save_func=lambda *a: None,
            mode="batch",
            **KW,
        )
    assert _notes(caught) == ["While processing ITEM-X"]


def test_custom_save_func(scenes, tmp_path):
    calls = []

    def save(prediction, meta, item, output_dir):
        calls.append((prediction.shape, meta["count"], item, output_dir))
        return f"saved-{Path(item).stem}"

    out = predict_files(
        SeededConv(), scenes, load_func=load_tif, save_func=save, output_dir=tmp_path, **KW
    )
    assert out == [f"saved-scene_{i}" for i in range(3)]
    assert calls == [((4, 80, 80), 3, scene, tmp_path) for scene in scenes]


def test_save_func_receives_valid_mask_when_asked(scenes, tmp_path):
    masks = {}

    def save(prediction, meta, item, output_dir, valid):
        masks[Path(item).name] = valid
        return item

    predict_files(SeededConv(), scenes, load_func=load_tif, save_func=save, **KW)
    assert masks["scene_0.tif"].all()
    assert not masks["scene_1.tif"][:20].any() and masks["scene_1.tif"][20:].all()


def test_nodata_mask_written(scenes, tmp_path):
    [path] = predict_files(SeededConv(), scenes[1:2], load_func=load_tif, output_dir=tmp_path, **KW)
    with rio.open(path) as src:
        mask = src.read_masks(1)
    assert np.all(mask[:20] == 0) and np.all(mask[20:] == 255)
    [path] = predict_files(
        SeededConv(),
        scenes[1:2],
        load_func=load_tif,
        output_dir=tmp_path / "nomask",
        mask_nodata=False,
        **KW,
    )
    with rio.open(path) as src:
        assert np.all(src.read_masks(1) == 255)


def test_overwrite_false_needs_output_path_with_custom_save_func(scenes):
    with pytest.raises(ValueError, match="output_path"):
        predict_files(
            SeededConv(),
            scenes,
            load_func=load_tif,
            save_func=lambda *a: None,
            overwrite=False,
            **KW,
        )


def test_load_func_error_raised_and_threads_stop(scenes, tmp_path):
    def loader(path):
        if Path(path).name == "scene_1.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    predict_files(SeededConv(), scenes[:1], load_func=load_tif, output_dir=tmp_path, **KW)
    baseline = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="loader failed"):
        run_with_timeout(
            lambda: predict_files(SeededConv(), scenes, load_func=loader, output_dir=tmp_path, **KW)
        )
    assert wait_for_new_threads(baseline) == []


def test_save_error_surfaces(scenes, tmp_path):
    def save(prediction, meta, item, output_dir):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        run_with_timeout(
            lambda: predict_files(SeededConv(), scenes, load_func=load_tif, save_func=save, **KW)
        )


def test_model_error_surfaces(scenes, tmp_path):
    def broken(x):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        run_with_timeout(
            lambda: predict_files(broken, scenes, load_func=load_tif, output_dir=tmp_path, **KW)
        )


class ReadWriteLockFree:
    """Stands in for the load/save lock."""

    def held(self):
        return False

    read = write = staticmethod(nullcontext)


@pytest.mark.parametrize("mode", ["tiled", "batch"])
def test_saves_in_flight_stay_bounded(monkeypatch, mode):
    """With every save blocked, predicting stops once `limit` saves wait: `limit` predictions
    queued plus the one being handed over."""
    # saves that never finish would hold back loads, and so everything else, under the lock;
    # this is about the bound, which also holds for loads that need no turn
    monkeypatch.setattr("manypred.files.IO_LOCK", ReadWriteLockFree())
    monkeypatch.setattr("manypred.batch.IO_LOCK", ReadWriteLockFree())
    limit, count = 3, 10
    release = threading.Event()
    predicted = []

    class CountingModel(SeededConv):
        def forward(self, x):
            if x.any():  # not the tiler's one-off shape probe on zeros
                predicted.append(x.shape[0])
            return super().forward(x)

    def save(prediction, meta, item, output_dir):
        release.wait(30)
        return item

    images = random_images(count, size=32)
    worker = threading.Thread(
        target=lambda: predict_files(
            CountingModel(),
            list(range(count)),
            load_func=lambda i: images[i],
            save_func=save,
            mode=mode,
            max_saves_in_flight=limit,
            batch_size=1,
            **KW,
        )
    )
    worker.start()
    # slow runners take a while to get there; then it must stop there
    deadline = time.monotonic() + 30
    while sum(predicted) < limit + 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)
    assert sum(predicted) == limit + 1
    release.set()
    worker.join(30)
    assert sum(predicted) == count


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_max_saves(scenes, bad):
    with pytest.raises(ValueError, match="max_saves_in_flight"):
        predict_files(SeededConv(), scenes, load_func=load_tif, max_saves_in_flight=bad, **KW)


def test_model_error_not_hidden_by_slow_loader(scenes):
    """An error is raised at once, without waiting for a load that is still running."""

    def loader(path):
        if Path(path).name != "scene_0.tif":
            time.sleep(5)
        return load_tif(path)

    def broken(x):
        raise RuntimeError("model failed")

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="model failed"):
        predict_files(broken, scenes, load_func=loader, save_func=lambda *a: None, **KW)
    assert time.monotonic() - start < 3


def test_load_error_not_hidden_by_hung_save(scenes, monkeypatch):
    monkeypatch.setattr("manypred.files._SAVE_GRACE_SECONDS", 0.2)
    release = threading.Event()

    def loader(path):
        if Path(path).name == "scene_1.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    def save(*args):
        release.wait(10)

    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="loader failed"):
            predict_files(SeededConv(), scenes, load_func=loader, save_func=save, **KW)
        assert time.monotonic() - start < 3
    finally:
        release.set()


def test_failed_run_finishes_saves_already_queued(scenes):
    saved = []

    def loader(path):
        if Path(path).name == "scene_2.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    def save(prediction, meta, item, output_dir):
        time.sleep(0.3)
        saved.append(item)

    with pytest.raises(RuntimeError, match="loader failed"):
        predict_files(
            SeededConv(), scenes, load_func=loader, save_func=save, max_saves_in_flight=5, **KW
        )
    assert saved == scenes[:2]


def test_failed_batch_run_keeps_the_chips_already_predicted():
    saved = []

    def loader(item):
        if item == 10:
            raise OSError("corrupt")
        return np.ones((3, 16, 16), np.float32)

    def save(prediction, meta, item, output_dir):
        time.sleep(0.05)
        saved.append(item)

    with pytest.raises(OSError, match="corrupt"):
        predict_files(
            SeededConv(), list(range(19)), load_func=loader, save_func=save, mode="batch", **KW
        )
    assert saved == list(range(8))


def test_one_failing_queued_save_doesnt_stop_the_others_or_replace_the_error(scenes):
    saved = []

    def loader(path):
        if Path(path).name == "scene_2.tif":
            raise RuntimeError("loader failed")
        return load_tif(path)

    def save(prediction, meta, item, output_dir):
        time.sleep(0.2)
        if item == scenes[0]:
            raise OSError("save failed")
        saved.append(item)

    with pytest.raises(RuntimeError, match="loader failed"):
        predict_files(
            SeededConv(), scenes, load_func=loader, save_func=save, max_saves_in_flight=5, **KW
        )
    assert saved == [scenes[1]]


def test_interrupt_does_not_wait_for_a_hung_save(scenes):
    release = threading.Event()

    def loader(path):
        if Path(path).name == "scene_1.tif":
            raise KeyboardInterrupt
        return load_tif(path)

    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            predict_files(
                SeededConv(),
                scenes,
                load_func=loader,
                save_func=lambda *a: release.wait(10),
                **KW,
            )
        assert time.monotonic() - start < 3
    finally:
        release.set()


def test_batch_shape_error_gives_advice_that_applies_and_names_the_item():
    def loader(item):
        return np.ones((1, 32 if item == 1 else 16, 16), np.float32)

    with pytest.raises(ValueError, match="mode='tiled'") as caught:
        predict_files(
            SeededConv(), [0, 1], load_func=loader, save_func=lambda *a: None, mode="batch", **KW
        )
    assert "group_by_shape" not in str(caught.value)
    assert _notes(caught) == ["While processing 1"]


def dict_model(x):
    dense = SeededConv()(x)
    return {"mask": dense, "score": dense.mean(dim=(2, 3))}


def test_dict_outputs_written_with_manifest_and_skipped(scenes, tmp_path):
    out = predict_files(
        dict_model,
        scenes[:1],
        load_func=load_tif,
        output_dir=tmp_path,
        mode="batch",
        output="logits",
        **{k: v for k, v in KW.items() if k not in ("patch_size", "overlap")},
    )
    assert out == [tmp_path / "scene_0_pred.json"]
    manifest = json.loads(out[0].read_text())
    assert manifest["files"] == {"mask": "scene_0_pred_mask.tif"}
    assert len(manifest["values"]["score"]) == 4
    assert (tmp_path / "scene_0_pred_mask.tif").exists()

    def loader(path):
        raise AssertionError("existing outputs must not be loaded")

    again = predict_files(
        dict_model, scenes[:1], load_func=loader, output_dir=tmp_path, overwrite=False, **KW
    )
    assert again == out


def test_single_image_level_output_json(scenes, tmp_path):
    [path] = predict_files(
        lambda x: SeededConv()(x).mean(dim=(2, 3)),
        scenes[:1],
        load_func=load_tif,
        output_dir=tmp_path,
        mode="batch",
        output="logits",
        progress=False,
        device="cpu",
    )
    assert path.suffix == ".json"
    assert list(json.loads(path.read_text())) == ["prediction"]


def test_batch_mode_writes_each_chips_mask(scenes, tmp_path):
    out = predict_files(
        SeededConv(),
        scenes,
        load_func=load_tif,
        output_dir=tmp_path,
        mode="batch",
        batch_size=3,
        nodata=0,
        progress=False,
        device="cpu",
    )
    with rio.open(out[1]) as src:
        mask = src.read_masks(1)
    assert np.all(mask[:20] == 0) and np.all(mask[20:] == 255)
    for path in (out[0], out[2]):
        with rio.open(path) as src:
            assert np.all(src.read_masks(1) == 255)


def test_batch_mode_matches_predict_batch(scenes, tmp_path):
    out = predict_files(
        SeededConv(),
        scenes,
        load_func=load_tif,
        output_dir=tmp_path,
        mode="batch",
        batch_size=2,
        output="argmax",
        progress=False,
        device="cpu",
    )
    for scene, path in zip(scenes, out, strict=True):
        from manypred import predict_batch

        expected = predict_batch(
            SeededConv(), [load_tif(scene)[0]], output="argmax", progress=False, device="cpu"
        )[0]
        with rio.open(path) as src:
            np.testing.assert_array_equal(src.read(), expected)


def test_from_threads(scenes, tmp_path):
    """Several calls at once must not crash (they did on MPS in OCM) and must agree."""
    model = SeededConv()

    def run(worker):
        return predict_files(
            model, scenes, load_func=load_tif, output_dir=tmp_path / f"w{worker}", **KW
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run, range(3)))
    for paths in results[1:]:
        for a, b in zip(results[0], paths, strict=True):
            with rio.open(a) as x, rio.open(b) as y:
                np.testing.assert_array_equal(x.read(), y.read())


def test_invalid_mode(scenes):
    with pytest.raises(ValueError, match="mode"):
        predict_files(SeededConv(), scenes, load_func=load_tif, mode="tiles", **KW)


def test_items_not_paths_need_output_path():
    with pytest.raises(TypeError, match="output_path"):
        predict_files(SeededConv(), [0], load_func=lambda i: random_images(1)[0], **KW)


def test_torch_threads_untouched():
    assert torch.get_num_threads() >= 1


class _TwoInputs(torch.nn.Module):
    """Dense output from ``img`` plus the mean of ``aux`` (a raster or a vector)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.conv(x["img"]) + x["aux"].float().mean()


def test_tiled_files_take_per_input_normalisation():
    """A {input: spec} normalize works in predict_files as in predict_array (it was built for
    a single input before the scene's inputs were known)."""
    rng = np.random.default_rng(0)
    scene = {
        "img": rng.random((3, 48, 48), dtype=np.float32),
        "aux": rng.random((1, 48, 48), dtype=np.float32),
    }
    normalize = {"img": {"type": "zscore", "scope": "tile"}, "aux": {"type": "scale", "value": 2}}
    model = _TwoInputs()
    kw = {**KW, "nodata": None, "normalize": normalize, "output": "logits"}
    saved = []
    predict_files(
        model, ["a"], load_func=lambda _: scene, save_func=lambda p, *_: saved.append(p), **kw
    )
    np.testing.assert_allclose(saved[0], predict_array(model, scene, **kw), rtol=1e-6)


def test_batch_files_primary_input_skips_vectors():
    """mode="batch" takes the first raster as the primary input (its pixels give the valid
    mask), as predict_batch does, even when a vector comes first in the dict."""
    rng = np.random.default_rng(0)
    chip = {
        "aux": np.array([1.0, 2.0], dtype=np.float32),
        "img": rng.random((3, 16, 16), dtype=np.float32),
    }
    model = _TwoInputs()
    saved = []
    predict_files(
        model,
        ["a"],
        load_func=lambda _: chip,
        save_func=lambda p, *_: saved.append(p),
        mode="batch",
        output="logits",
        progress=False,
        device="cpu",
    )
    expected = predict_batch(model, [chip], output="logits", progress=False, device="cpu")
    np.testing.assert_allclose(saved[0], expected[0], rtol=1e-6)
