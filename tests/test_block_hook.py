"""BlockPostprocess: block post-processing while predict_files predicts."""

import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from helpers import SeededConv

from manypred import BlockPostprocess, postprocess_blocks, predict_array, predict_files
from manypred.blocks import BlockStream, Scene, block_grid
from manypred.geo import RasterioReader

rio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402

KW = {"device": "cpu", "progress": False}
RNG = np.random.default_rng(0)
TRANSFORM = Affine(0.5, 0, 400000, 0, -0.5, 6500000)


def windows(fields, block, scene):
    """Module level (worker processes import it): the block's windows, as they came."""
    return block.index, fields


def same(a, b):
    """Equal results of :func:`windows` (NaN where nothing was predicted)."""
    assert len(a) == len(b)
    for (i, x), (j, y) in zip(a, b, strict=True):
        assert i == j and set(x) == set(y)
        for name in x:
            assert x[name].dtype == y[name].dtype == np.float32
            np.testing.assert_array_equal(x[name], y[name])


@pytest.mark.parametrize(
    "height, width, block, halo",
    [(300, 170, 64, 16), (97, 50, 10, 30), (40, 40, 100, 8), (61, 23, 7, 0), (1, 5, 4, 2)],
)
@pytest.mark.parametrize("seed", range(3))
def test_stream_equals_postprocess_blocks(height, width, block, halo, seed):
    rng = np.random.default_rng(seed)
    fields = {
        "a": rng.random((2, height, width)).astype(np.float32),
        "b": rng.random((1, height, width)).astype(np.float32),
    }
    scene = Scene(height, width)
    post = BlockPostprocess(windows, block=block, halo=halo)
    stream = post.stream(scene)
    stream.select({"a": 1, "b": 1, "c": None})
    top = 0
    while top < height:  # rows in random chunks, as BlockSink finishes them
        rows = int(rng.integers(1, 40))
        stream.rows(top, {name: x[:, top : top + rows] for name, x in fields.items()})
        top += rows
    same(stream.close(), postprocess_blocks(fields, windows, block=block, halo=halo))
    assert stream.capacity <= min(height, block + 2 * halo)
    for buffer in stream.buffers.values():
        assert buffer.shape[1] == stream.capacity


def test_stream_runs_each_strip_as_soon_as_its_rows_are_in(monkeypatch):
    runs = []
    real = BlockStream._run_final

    def recording(self):
        before = self.next
        real(self)
        runs.extend([self.end] * (self.next - before))

    monkeypatch.setattr(BlockStream, "_run_final", recording)
    stream = BlockPostprocess(windows, block=50, halo=10).stream(Scene(1000, 30))
    stream.select({"a": 1})
    x = RNG.random((1, 1000, 30)).astype(np.float32)
    for top in range(0, 1000, 7):
        stream.rows(top, {"a": x[:, top : top + 7]})
    expected = [min(b.window[1], 1000) for b in block_grid(1000, 30, 50, 10)]
    assert runs == expected  # one block per strip here
    assert stream.capacity == 70  # a strip plus two halos, not the scene


class Heads(torch.nn.Module):
    """Two full-resolution heads that differ between patches (so cut and blend differ), a
    quarter-resolution head and an image-level one."""

    def forward(self, x):
        patch = x.mean(dim=(1, 2, 3), keepdim=True)
        extent = SeededConv()(x)[:, :2] + patch
        return {
            "extent": extent,
            "dist": x[:, :1] * 2 - patch,
            "quarter": F.avg_pool2d(extent, 4),
            "score": x.mean(dim=(2, 3)),
        }


HEAD_KW = {
    "activation": {"extent": "sigmoid", "score": "sigmoid"},
    "output": {"score": "probs"},
    "patch_size": 48,
    "overlap": 16,
    "nodata": 0,
}


def two_scenes(path):
    """``path`` and a copy of it: two items with their own outputs."""
    copy = path.with_name(f"{path.stem}_copy{path.suffix}")
    shutil.copy(path, copy)
    return [path, copy]


def write_scene(tmp_path, height=300, width=170):
    x = (RNG.random((3, height, width)) * 1000 + 1).astype(np.uint16)
    x[:, :30, :60] = 0  # nodata: masked, and patches shifted
    x[:, -50:, -40:] = 0
    path = tmp_path / "scene.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32750",
        "transform": TRANSFORM,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(x)
    return path


def load_lazy(path):
    reader = RasterioReader(path)
    return reader, reader.profile


def load_eager(path):
    with rio.open(path) as src:
        return src.read(), src.profile


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("merge", ["cut", "blend"])
def test_hook_equals_postprocess_blocks_over_the_written_rasters(
    tmp_path, lazy, workers, merge, monkeypatch
):
    path = write_scene(tmp_path)
    streamed = []
    rows = BlockStream.rows
    monkeypatch.setattr(
        BlockStream, "rows", lambda self, top, f: streamed.append(top) or rows(self, top, f)
    )
    post = BlockPostprocess(windows, block=64, halo=16, workers=workers, fields=["extent", "dist"])
    [manifest] = predict_files(
        Heads(),
        [path],
        load_func=load_lazy if lazy else load_eager,
        output_dir=tmp_path / "out",
        merge=merge,
        block_postprocess=post,
        **HEAD_KW,
        **KW,
    )
    assert manifest.name == "scene_pred.json"  # predict_files returns what it always did
    assert (len(streamed) > 3) == lazy  # lazy scenes stream their rows, eager ones don't
    rasters = {name: tmp_path / "out" / f"scene_pred_{name}.tif" for name in ("extent", "dist")}
    expected = postprocess_blocks(rasters, windows, block=64, halo=16)
    assert len(expected) == 5 * 3  # several block rows
    same(post.results[0], expected)


def test_cut_and_blend_hooks_differ(tmp_path):
    path = write_scene(tmp_path)
    found = []
    for merge in ("cut", "blend"):
        post = BlockPostprocess(windows, block=64, halo=16, fields=["extent"])
        predict_files(
            Heads(),
            [path],
            load_func=load_lazy,
            output_dir=tmp_path / merge,
            merge=merge,
            block_postprocess=post,
            **HEAD_KW,
            **KW,
        )
        found.append(post.results[0][4][1]["extent"])
    assert not np.array_equal(found[0], found[1], equal_nan=True)


def test_lazy_and_eager_hooks_agree(tmp_path):
    path = write_scene(tmp_path)
    results = []
    for load in (load_lazy, load_eager):
        post = BlockPostprocess(windows, block=64, halo=16, fields=["extent", "dist"])
        predict_files(
            Heads(),
            [path],
            load_func=load,
            output_dir=tmp_path / load.__name__,
            block_postprocess=post,
            merge="cut",
            **HEAD_KW,
            **KW,
        )
        results.append(post.results[0])
    same(*results)


def scene_of(fields, block, scene):
    return scene


@pytest.mark.parametrize("load", [load_lazy, load_eager])
def test_save_combine_and_scene(tmp_path, load):
    path = write_scene(tmp_path)
    saved = []

    def combine(results, scene):
        return {"blocks": len(results), "scene": scene, "same": all(s == scene for s in results)}

    def save(result, item, scene):
        saved.append((result, item, scene))
        return "saved"

    post = BlockPostprocess(
        scene_of, block=100, halo=8, combine=combine, save=save, fields=["extent"]
    )
    predict_files(
        Heads(),
        two_scenes(path),
        load_func=load,
        output_dir=tmp_path / "out",
        block_postprocess=post,
        **HEAD_KW,
        **KW,
    )
    assert post.results == {0: "saved", 1: "saved"}
    assert len(saved) == 2 and saved[0][1] == path
    result, _, scene = saved[0]
    assert scene == Scene(300, 170, TRANSFORM, scene.crs) and scene.crs.to_epsg() == 32750
    assert result == {"blocks": 6, "scene": scene, "same": True}


@pytest.mark.parametrize("load", [load_lazy, load_eager])
def test_hook_sees_probabilities_not_classes(tmp_path, load):
    path = write_scene(tmp_path)
    kwargs = {"patch_size": 48, "overlap": 16, "nodata": 0, "activation": "softmax", **KW}
    post = BlockPostprocess(windows, block=64, halo=16)
    predict_files(
        SeededConv(),
        [path],
        load_func=load,
        output_dir=tmp_path / "out",
        output="argmax",
        block_postprocess=post,
        **kwargs,
    )
    with rio.open(tmp_path / "out" / "scene_pred.tif") as src:
        assert src.dtypes[0] == "uint8"  # classes saved
    image, _ = load_eager(path)
    probs = predict_array(SeededConv(), image, output="probs", **kwargs)
    same(post.results[0], postprocess_blocks({"": probs}, windows, block=64, halo=16))


def test_hook_ignores_postprocess(tmp_path):
    path = write_scene(tmp_path)
    plain = BlockPostprocess(windows, block=64, halo=16, fields=["dist"])
    after = BlockPostprocess(windows, block=64, halo=16, fields=["dist"])
    common = {"output_dir": tmp_path / "out", **HEAD_KW, **KW}
    predict_files(Heads(), [path], load_func=load_lazy, block_postprocess=plain, **common)
    predict_files(
        Heads(),
        [path],
        load_func=load_lazy,
        block_postprocess=after,
        postprocess=lambda result: {name: x * 0 for name, x in result.items() if name != "score"},
        **common,
    )
    same(after.results[0], plain.results[0])


def test_skipped_items_are_not_post_processed(tmp_path):
    path = write_scene(tmp_path)
    common = {"load_func": load_lazy, "output_dir": tmp_path / "out", **HEAD_KW, **KW}
    predict_files(Heads(), [path], **common)
    post = BlockPostprocess(windows, fields=["dist"])
    predict_files(Heads(), [path], overwrite=False, block_postprocess=post, **common)
    assert post.results == {}


@pytest.mark.parametrize("load", [load_lazy, load_eager])
def test_hook_field_errors(tmp_path, load):
    path = write_scene(tmp_path)
    common = {"load_func": load, "output_dir": tmp_path / "out", **HEAD_KW, **KW}
    with pytest.raises(ValueError, match=r"reduced-resolution heads.*\['extent', 'dist'\]"):
        predict_files(Heads(), [path], block_postprocess=BlockPostprocess(windows), **common)
    with pytest.raises(ValueError, match="image-level"):
        post = BlockPostprocess(windows, fields=["extent", "score"])
        predict_files(Heads(), [path], block_postprocess=post, **common)
    with pytest.raises(ValueError, match="not outputs of the model"):
        post = BlockPostprocess(windows, fields=["nope"])
        predict_files(Heads(), [path], block_postprocess=post, **common)


def test_hook_arguments():
    with pytest.raises(ValueError, match="block"):
        BlockPostprocess(windows, block=0)
    with pytest.raises(ValueError, match="halo"):
        BlockPostprocess(windows, halo=-1)
    with pytest.raises(ValueError, match="workers"):
        BlockPostprocess(windows, workers=-1)
    with pytest.raises(ValueError, match="mode='tiled'"):
        predict_files(
            SeededConv(),
            [],
            load_func=lambda p: p,
            mode="batch",
            block_postprocess=BlockPostprocess(windows),
            **KW,
        )


def fail(fields, block, scene):
    raise RuntimeError("decoder failed")


@pytest.mark.parametrize("workers", [0, 2])
def test_hook_errors_propagate(tmp_path, workers):
    path = write_scene(tmp_path)
    post = BlockPostprocess(fail, block=64, halo=16, workers=workers, fields=["dist"])
    with pytest.raises(RuntimeError, match="decoder failed"):
        predict_files(
            Heads(),
            [path],
            load_func=load_lazy,
            output_dir=tmp_path / "out",
            block_postprocess=post,
            **HEAD_KW,
            **KW,
        )


def fail_late(fields, block, scene):
    if block.index == 14:  # the last block
        raise RuntimeError("decoder failed")
    return block.index


@pytest.mark.parametrize("load", [load_lazy, load_eager])
@pytest.mark.parametrize(
    "where, workers",
    [("fn", 0), ("fn", 2), ("combine", 0), ("save", 0)],  # spawning workers takes seconds
)
def test_failed_post_processing_leaves_the_item_undone(tmp_path, load, workers, where):
    # the rasters' manifest marks an item done for overwrite=False, so it must not be written
    # when the block post-processing of that item failed
    path = write_scene(tmp_path)

    def bad(*args):
        raise RuntimeError("decoder failed")

    post = BlockPostprocess(
        fail_late if where == "fn" else windows,
        block=64,
        halo=16,
        workers=workers,
        fields=["dist"],
        combine=bad if where == "combine" else None,
        save=bad if where == "save" else None,
    )
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="decoder failed"):
        predict_files(
            Heads(), [path], load_func=load, output_dir=out, block_postprocess=post, **HEAD_KW, **KW
        )
    assert not (out / "scene_pred.json").exists()
    if load is load_lazy:
        assert not list(out.glob("*.tif"))  # partial rasters removed


def threads(fields, block, scene):
    return torch.get_num_threads()


def test_hook_pool_is_shared_by_scenes_and_closed(tmp_path, monkeypatch):
    from helpers import run_with_timeout

    import manypred.blocks as blocks

    made = []
    real = blocks.make_pool
    monkeypatch.setattr(blocks, "make_pool", lambda n: made.append(n) or real(n))
    path = write_scene(tmp_path)
    torch.ones(64, 64) @ torch.ones(64, 64)  # torch used here first
    for load in (load_lazy, load_eager):
        post = BlockPostprocess(threads, block=64, halo=16, workers=2, fields=["dist"])
        run_with_timeout(
            lambda post=post, load=load: predict_files(
                Heads(),
                two_scenes(path),
                load_func=load,
                output_dir=tmp_path / load.__name__,
                block_postprocess=post,
                **HEAD_KW,
                **KW,
            ),
            timeout=120,
        )
        assert set(post.results) == {0, 1}
        assert {n for r in post.results.values() for n in r} == {1}
        assert post._executor is None  # closed by predict_files
    assert made == [2, 2]  # one pool per call, both scenes in it


def test_pool_closed_when_a_save_fails(tmp_path):
    path = write_scene(tmp_path)
    post = BlockPostprocess(threads, block=64, halo=16, workers=2, fields=["dist"])

    def failing_save(prediction, meta, item, output_dir):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        predict_files(
            Heads(),
            [path],
            load_func=load_eager,
            save_func=failing_save,
            block_postprocess=post,
            **HEAD_KW,
            **KW,
        )
    assert post._executor is None


def test_results_are_per_call(tmp_path):
    path = write_scene(tmp_path)
    post = BlockPostprocess(scene_of, block=100, halo=8, fields=["dist"])
    common = {
        "load_func": load_lazy,
        "output_dir": tmp_path / "out",
        "block_postprocess": post,
        **HEAD_KW,
        **KW,
    }
    predict_files(Heads(), two_scenes(path), **common)
    assert set(post.results) == {0, 1}
    predict_files(Heads(), [path], overwrite=False, **common)  # skipped: nothing post-processed
    assert post.results == {}
