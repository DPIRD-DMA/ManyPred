"""manypred.blocks: the block grid and block post-processing over stitched rasters."""

import os
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from manypred.blocks import Block, Scene, block_grid, postprocess_blocks

RNG = np.random.default_rng(0)

GRIDS = [
    (1, 1, 4, 2),
    (7, 5, 3, 1),
    (64, 64, 64, 8),  # one block exactly
    (50, 70, 100, 16),  # block larger than the scene
    (97, 211, 40, 13),  # nothing divides
    (100, 130, 32, 0),  # no halo
    (33, 17, 5, 50),  # halo larger than the scene
]


@pytest.mark.parametrize("height, width, size, halo", GRIDS)
def test_cores_partition_the_scene(height, width, size, halo):
    blocks = block_grid(height, width, size, halo)
    count = np.zeros((height, width), dtype=int)
    for i, b in enumerate(blocks):
        assert b.index == i
        top, bottom, left, right = b.core
        assert 0 <= top < bottom <= height and 0 <= left < right <= width
        assert bottom - top <= size and right - left <= size
        count[top:bottom, left:right] += 1
    np.testing.assert_array_equal(count, 1)
    # row-major
    assert [b.core[:3:2] for b in blocks] == sorted(b.core[:3:2] for b in blocks)
    assert len(blocks) == -(-height // size) * -(-width // size)


@pytest.mark.parametrize("height, width, size, halo", GRIDS)
def test_windows_are_cores_grown_by_the_halo_and_clipped(height, width, size, halo):
    for b in block_grid(height, width, size, halo):
        top, bottom, left, right = b.core
        assert b.window == (
            max(top - halo, 0),
            min(bottom + halo, height),
            max(left - halo, 0),
            min(right + halo, width),
        )
        assert b.scene_edges == (
            b.window[0] == 0,
            b.window[1] == height,
            b.window[2] == 0,
            b.window[3] == width,
        )
        t, bt, lft, r = b.inner
        assert (t + b.window[0], bt + b.window[0], lft + b.window[2], r + b.window[2]) == b.core


def test_scene_edges_of_a_three_by_three_grid():
    blocks = block_grid(30, 30, 10, 2)
    assert blocks[0].scene_edges == (True, False, True, False)
    assert blocks[4].scene_edges == (False, False, False, False)
    assert blocks[4].window == (8, 22, 8, 22)
    assert blocks[8].scene_edges == (False, True, False, True)
    # a halo reaching the edge makes it a scene edge
    assert block_grid(30, 30, 10, 10)[4].scene_edges == (True, True, True, True)


def test_grid_arguments():
    with pytest.raises(ValueError, match="block"):
        block_grid(10, 10, 0)
    with pytest.raises(ValueError, match="halo"):
        block_grid(10, 10, 4, -1)
    assert block_grid(0, 10, 4) == []


def windows_of(fields, block, scene):
    """Each field's window, its block and its scene."""
    return {name: x.copy() for name, x in fields.items()}, block, scene


def core_of(fields, block, scene):
    t, b, lft, r = block.inner
    return {name: x[:, t:b, lft:r] for name, x in fields.items()}


def assemble(results, scene):
    """Cores put back together on the scene grid."""
    out = {}
    for block, cores in zip(block_grid(scene.height, scene.width, 16, 5), results, strict=True):
        for name, core in cores.items():
            array = out.setdefault(
                name, np.zeros((core.shape[0], scene.height, scene.width), dtype=core.dtype)
            )
            top, bottom, left, right = block.core
            array[:, top:bottom, left:right] = core
    return out


def fields_for(height=45, width=58):
    return {
        "probs": RNG.random((2, height, width)).astype(np.float32),
        "mask": (RNG.random((height, width)) > 0.5).astype(np.uint8),  # 2-D
    }


@pytest.mark.parametrize("height, width", [(45, 58), (16, 16), (3, 70), (1, 1)])
def test_windows_are_3d_crops_in_the_stored_dtype(height, width):
    fields = fields_for(height, width)
    results = postprocess_blocks(fields, windows_of, block=16, halo=5)
    blocks = block_grid(height, width, 16, 5)
    assert len(results) == len(blocks)
    for (window, block, scene), expected in zip(results, blocks, strict=True):
        assert block == expected
        assert scene == Scene(height, width)
        top, bottom, left, right = block.window
        np.testing.assert_array_equal(window["probs"], fields["probs"][:, top:bottom, left:right])
        np.testing.assert_array_equal(window["mask"], fields["mask"][None, top:bottom, left:right])
        assert window["probs"].dtype == np.float32 and window["mask"].dtype == np.uint8
        assert window["mask"].ndim == 3


def test_fn_gets_copies():
    fields = fields_for()
    before = {name: x.copy() for name, x in fields.items()}

    def scribble(window, block, scene):
        for x in window.values():
            x[...] = 0

    postprocess_blocks(fields, scribble, block=16, halo=5)
    for name in fields:
        np.testing.assert_array_equal(fields[name], before[name])


def test_combine_gets_results_in_grid_order():
    fields = fields_for()
    calls = []

    def combine(results, scene):
        calls.append(scene)
        return assemble(results, scene)

    out = postprocess_blocks(fields, core_of, block=16, halo=5, combine=combine)
    assert calls == [Scene(45, 58)]
    np.testing.assert_array_equal(out["probs"], fields["probs"])
    np.testing.assert_array_equal(out["mask"], fields["mask"][None])


def test_memmap_and_given_scene(tmp_path):
    array = np.lib.format.open_memmap(
        tmp_path / "f.npy", mode="w+", dtype=np.float32, shape=(1, 40, 30)
    )
    array[:] = RNG.random((1, 40, 30))
    scene = Scene(40, 30, transform="t", crs="c")
    results = postprocess_blocks({"f": array}, windows_of, block=16, halo=5, scene=scene)
    assert all(s is scene for _, _, s in results)
    assert type(results[0][0]["f"]) is np.ndarray
    with pytest.raises(ValueError, match="scene is 41 x 30"):
        postprocess_blocks({"f": array}, windows_of, scene=Scene(41, 30))


def test_field_errors():
    with pytest.raises(ValueError, match="same height and width"):
        postprocess_blocks({"a": np.zeros((5, 6)), "b": np.zeros((1, 5, 7))}, windows_of)
    with pytest.raises(ValueError, match=r"\(C, H, W\) or \(H, W\)"):
        postprocess_blocks({"a": np.zeros((1, 1, 5, 6))}, windows_of)
    with pytest.raises(ValueError, match="at least one field"):
        postprocess_blocks({}, windows_of)
    with pytest.raises(TypeError, match="array-like"):
        postprocess_blocks({"a": [[1, 2]]}, windows_of)
    with pytest.raises(ValueError, match="workers"):
        postprocess_blocks({"a": np.zeros((5, 6))}, windows_of, workers=-1)


def write_tif(path, array, transform=None, crs=None):
    rio = pytest.importorskip("rasterio")
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    if transform is not None:
        profile.update(transform=transform, crs=crs)
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path


def test_lazy_geotiffs_equal_in_memory(tmp_path):
    pytest.importorskip("rasterio")
    from affine import Affine

    fields = fields_for(70, 90)
    transform = Affine(0.5, 0, 400000, 0, -0.5, 6500000)
    paths = {
        "probs": write_tif(tmp_path / "probs.tif", fields["probs"], transform, "EPSG:32750"),
        "mask": str(
            write_tif(tmp_path / "mask.tif", fields["mask"][None], transform, "EPSG:32750")
        ),
    }
    expected = postprocess_blocks(fields, windows_of, block=16, halo=5)
    lazy = postprocess_blocks(paths, windows_of, block=16, halo=5)
    assert len(lazy) == len(expected)
    for (a, block_a, _), (b, block_b, _) in zip(expected, lazy, strict=True):
        assert block_a == block_b
        for name in fields:
            assert a[name].dtype == b[name].dtype
            np.testing.assert_array_equal(a[name], b[name])
    scene = lazy[0][2]
    assert scene.transform == transform and scene.crs.to_epsg() == 32750
    assert (scene.height, scene.width) == (70, 90)
    # mixed: an array and a file, the scene from the file
    mixed = postprocess_blocks(
        {"probs": fields["probs"], "mask": paths["mask"]}, windows_of, block=16, halo=5
    )
    assert mixed[0][2].transform == transform
    np.testing.assert_array_equal(mixed[-1][0]["mask"], expected[-1][0]["mask"])


def total(fields, block, scene):
    """Module level, so worker processes can import it."""
    t, b, lft, r = block.inner
    return block.index, {name: float(x[:, t:b, lft:r].sum()) for name, x in fields.items()}


def test_workers_give_the_same_results_in_order(tmp_path):
    pytest.importorskip("rasterio")
    fields = fields_for(70, 90)
    paths = {"probs": write_tif(tmp_path / "probs.tif", fields["probs"]), "mask": fields["mask"]}
    serial = postprocess_blocks(paths, total, block=16, halo=5)
    parallel = postprocess_blocks(paths, total, block=16, halo=5, workers=2)
    assert parallel == serial
    assert [index for index, _ in parallel] == list(range(len(block_grid(70, 90, 16))))
    assert sum(r["mask"] for _, r in serial) == fields["mask"].sum()


def fail_on_third(fields, block, scene):
    if block.index == 2:
        raise RuntimeError("block failed")
    return block.index


def test_worker_errors_propagate():
    with pytest.raises(RuntimeError, match="block failed"):
        postprocess_blocks({"a": np.zeros((40, 40))}, fail_on_third, block=8, workers=2)


def components(mask):
    """Connected components (4-connected) as sets of flat pixel indices, by flood fill."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for r, c in zip(*np.nonzero(mask), strict=True):
        if seen[r, c]:
            continue
        pixels, queue = [], deque([(r, c)])
        seen[r, c] = True
        while queue:
            y, x = queue.popleft()
            pixels.append(y * width + x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        found.append(frozenset(pixels))
    return found


def owned_components(fields, block: Block, scene: Scene):
    """Components of the window whose anchor (first pixel in raster order) is in the core and
    that don't touch a cut edge of the window, in scene pixel indices."""
    mask = fields["mask"][0] > 0
    top, bottom, left, right = block.window
    h, w = mask.shape
    owned = []
    for comp in components(mask):
        rows, cols = np.divmod(np.array(sorted(comp)), w)
        touches = (
            (rows.min() == 0 and not block.scene_edges[0])
            or (rows.max() == h - 1 and not block.scene_edges[1])
            or (cols.min() == 0 and not block.scene_edges[2])
            or (cols.max() == w - 1 and not block.scene_edges[3])
        )
        anchor = (rows[0] + top, cols[0] + left)
        core = block.core
        if touches or not (core[0] <= anchor[0] < core[1] and core[2] <= anchor[1] < core[3]):
            continue
        owned.append(frozenset((rows + top) * scene.width + cols + left))
    return owned


@pytest.mark.parametrize("size, halo", [(16, 8), (23, 8), (40, 10), (200, 0)])
def test_owned_components_equal_the_global_labelling(size, halo):
    # small blobs (at most 7 px across), so a halo of 8 holds any blob anchored in the core
    rng = np.random.default_rng(size)
    mask = np.zeros((75, 93), dtype=np.uint8)
    for _ in range(60):
        r, c = rng.integers(0, 75), rng.integers(0, 93)
        mask[r : r + rng.integers(1, 7), c : c + rng.integers(1, 7)] = 1
    expected = set(components(mask.astype(bool)))
    found = postprocess_blocks(
        {"mask": mask},
        owned_components,
        block=size,
        halo=halo,
        combine=lambda results, scene: [c for r in results for c in r],
    )
    assert len(found) == len(set(found))  # no object owned twice
    assert set(found) == expected


def test_memmaps_are_read_in_the_workers(tmp_path, monkeypatch):
    import manypred.blocks as blocks

    array = np.lib.format.open_memmap(
        tmp_path / "f.npy", mode="w+", dtype=np.float32, shape=(2, 70, 90)
    )
    array[:] = RNG.random((2, 70, 90))
    array.flush()
    view = array[:, 10:60]  # a view: cut here, it can't be reopened from its offset
    assert blocks._Memmap.of(array) is not None and blocks._Memmap.of(view) is None
    expected = postprocess_blocks({"f": np.asarray(array).copy()}, total, block=16, halo=5)
    cuts = []
    real = blocks._cut
    monkeypatch.setattr(blocks, "_cut", lambda x, w: cuts.append(w) or real(x, w))
    assert postprocess_blocks({"f": array}, total, block=16, halo=5, workers=2) == expected
    assert cuts == []  # nothing cut in this process
    sliced = postprocess_blocks({"f": view}, total, block=16, halo=5, workers=2)
    assert sliced == postprocess_blocks({"f": np.array(view)}, total, block=16, halo=5)
    assert cuts


def torch_block(fields, block, scene):
    """Torch on the CPU inside a worker: a convolution over the window, its core summed, and
    the worker's thread count."""
    import torch

    x = torch.from_numpy(fields["f"])[None]
    smoothed = torch.nn.functional.avg_pool2d(x, 3, 1, 1)[0]
    t, b, lft, r = block.inner
    return block.index, float(smoothed[:, t:b, lft:r].sum()), torch.get_num_threads()


def test_workers_with_torch_after_torch_ran_here():
    # torch's thread pool started in this process; forked workers then hang in torch (Python
    # 3.12 on Linux). Workers start with spawn, one thread each, and must finish in time.
    import torch
    from helpers import run_with_timeout

    torch.ones(256, 256) @ torch.ones(256, 256)
    fields = {"f": RNG.random((2, 90, 70)).astype(np.float32)}
    serial = postprocess_blocks(fields, torch_block, block=32, halo=4)
    parallel = run_with_timeout(
        lambda: postprocess_blocks(fields, torch_block, block=32, halo=4, workers=2), timeout=120
    )
    assert [r[:2] for r in parallel] == [r[:2] for r in serial]
    assert {r[2] for r in parallel} == {1}  # one compute thread per worker


def test_pools_start_with_spawn():
    from manypred.blocks import make_pool

    pool = make_pool(1)
    try:
        assert pool._mp_context.get_start_method() == "spawn"
    finally:
        pool.shutdown()


def start_environment(fields, block, scene):
    """The worker's environment as it started (before any of its imports)."""
    with open("/proc/self/environ", "rb") as f:
        pairs = [item.split(b"=", 1) for item in f.read().split(b"\0") if b"=" in item]
    return {key.decode(): value.decode() for key, value in pairs}


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
def test_workers_start_with_one_thread_variables(monkeypatch):
    # numpy (BLAS) is imported by a spawned worker before any initializer runs, so the
    # variables must be there when the process starts; this process's are left alone
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    [env] = postprocess_blocks({"f": np.zeros((4, 4))}, start_environment, block=8, workers=1)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "7" and "OPENBLAS_NUM_THREADS" not in os.environ


@pytest.mark.parametrize(
    "other",
    [
        {"transform": (10.0, 0.0, 400010.0, 0.0, -10.0, 6500000.0), "crs": "EPSG:32750"},
        {"transform": (10.0, 0.0, 400000.0, 0.0, -10.0, 6500000.0), "crs": "EPSG:32751"},
        {"transform": (10.0, 0.0, 400000.0, 0.0, -10.0, 6500000.0), "crs": None},
    ],
)
def test_raster_fields_must_share_a_grid(tmp_path, other):
    rio = pytest.importorskip("rasterio")
    from affine import Affine

    def tif(path, transform, crs):
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float32",
            transform=Affine(*transform),
            crs=crs,
        ) as dst:
            dst.write(np.zeros((1, 8, 8), np.float32))
        return path

    base = (10.0, 0.0, 400000.0, 0.0, -10.0, 6500000.0)
    a = tif(tmp_path / "a.tif", base, "EPSG:32750")
    b = tif(tmp_path / "b.tif", other["transform"], other["crs"])
    with pytest.raises(ValueError, match="must share a grid.*'b'"):
        postprocess_blocks({"a": a, "b": b}, lambda f, b, s: None, block=8, halo=0)
    same = tif(tmp_path / "c.tif", base, "EPSG:32750")
    assert postprocess_blocks({"a": a, "c": same}, lambda f, b, s: 1, block=8, halo=0) == [1]
