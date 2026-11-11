"""Loads and saves take turns (OmniCloudMask#52), and what files are written with."""

import os
import threading
import time

import numpy as np
import pytest
from helpers import SeededConv

from manypred import predict_files
from manypred.io_lock import IO_LOCK, ReadWriteLock

WAIT = 10


def scene(_=None):
    return np.random.default_rng(0).integers(1, 1000, size=(3, 64, 64), dtype=np.uint16)


KW = {"patch_size": 32, "overlap": 8, "device": "cpu", "progress": False, "nodata": 0}


def run_thread(target):
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def test_readers_share_and_a_writer_excludes_them():
    lock = ReadWriteLock()
    reading, wrote = threading.Event(), threading.Event()

    def other_reader():
        with lock.read():
            reading.set()

    def writer():
        with lock.write():
            wrote.set()

    with lock.read():
        run_thread(other_reader).join(WAIT)
        assert reading.is_set()
        thread = run_thread(writer)
        assert not wrote.wait(0.05)
    thread.join(WAIT)
    assert wrote.is_set()


def test_a_waiting_writer_holds_back_new_readers_but_not_the_reader_it_waits_for():
    lock = ReadWriteLock()
    events = []

    def writer():
        with lock.write():
            events.append("write")

    def late_reader():
        with lock.read():
            events.append("read")

    with lock.read():
        first = run_thread(writer)
        while not lock._writers_waiting:
            time.sleep(0.001)
        second = run_thread(late_reader)
        time.sleep(0.05)
        assert events == []
        with lock.read():  # re-entrant: this read must not wait for the writer
            pass
    first.join(WAIT)
    second.join(WAIT)
    assert events == ["write", "read"]


def test_saving_inside_a_load_raises_instead_of_hanging():
    lock = ReadWriteLock()
    with lock.read(), pytest.raises(RuntimeError, match="take turns"), lock.write():
        pass
    with lock.write():  # free again
        pass


def test_the_writer_may_read_and_write_again():
    lock = ReadWriteLock()
    with lock.write(), lock.read(), lock.write():
        assert lock.held()
    assert not lock.held()


def test_an_error_inside_frees_the_lock():
    lock = ReadWriteLock()
    for side in (lock.read, lock.write):
        with pytest.raises(ValueError), side():
            raise ValueError
    with lock.write():
        pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_starts_with_the_lock_free():
    with IO_LOCK.read():
        pid = os.fork()
        if pid == 0:
            status = 1
            try:
                with IO_LOCK.write():
                    status = 0
            finally:
                os._exit(status)
    assert os.waitpid(pid, 0)[1] == 0


def _overlaps(load_func_body=None):
    """A load that stays open until a save has asked for its turn or started, so without the
    lock the two always overlap. Returns (load_func, save_func, overlaps seen)."""
    state = {"loads": 0, "saves": 0}
    overlaps = []
    guard = threading.Lock()

    def load(item):
        with guard:
            state["loads"] += 1
        deadline = time.time() + WAIT
        while (
            item > 0
            and state["saves"] == 0
            and not IO_LOCK._writers_waiting
            and time.time() < deadline
        ):
            time.sleep(0.001)
        with guard:
            if state["saves"]:
                overlaps.append(("load during save", item))
            state["loads"] -= 1
        return scene()

    def save(prediction, meta, item, output_dir):
        with guard:
            if state["loads"]:
                overlaps.append(("save during load", item))
            state["saves"] += 1
        time.sleep(0.02)
        with guard:
            state["saves"] -= 1
        return item

    return load, save, overlaps


def test_loads_never_overlap_saves():
    load, save, overlaps = _overlaps()
    done = predict_files(SeededConv(), range(4), load_func=load, save_func=save, **KW)
    assert done == [0, 1, 2, 3]
    assert overlaps == []


def test_batch_mode_loads_never_overlap_saves():
    load, save, overlaps = _overlaps()
    done = predict_files(
        SeededConv(), range(4), load_func=load, save_func=save, mode="batch", device="cpu",
        progress=False, nodata=0, batch_size=2,
    )  # fmt: skip
    assert done == [0, 1, 2, 3]
    assert overlaps == []


class _Reader:
    shape, dtype = (3, 64, 64), np.dtype(np.uint16)

    def __init__(self, state):
        self.state = state

    def read(self, window):
        top, bottom, left, right = window
        assert not self.state["writing"], "read during a write"
        self.state["reads"] += 1
        return scene()[:, top:bottom, left:right]


class _BlockWriter:
    def __init__(self, state):
        self.state = state

    def write(self, name, top, block, shape, valid, factor=1):
        self.state["writing"] = True
        self.state["writes"] += 1
        assert IO_LOCK._writing
        time.sleep(0.001)
        self.state["writing"] = False

    def close(self, extras=None):
        assert IO_LOCK._writing
        return "done"

    def abort(self):
        assert IO_LOCK._writing
        self.state["aborted"] = True


class _Save:
    writes_blocks = True

    def __init__(self, state):
        self.state = state

    def open(self, meta, item, output_dir, grid=None):
        assert IO_LOCK._writing
        return _BlockWriter(self.state)

    def __call__(self, *args, **kwargs):
        raise AssertionError("lazy scenes go through open")


def test_lazy_reads_and_block_writes_take_turns():
    state = {"writing": False, "reads": 0, "writes": 0}
    done = predict_files(
        SeededConv(), [0, 1], load_func=lambda item: _Reader(state), save_func=_Save(state), **KW
    )
    assert done == ["done", "done"]
    assert state["reads"] and state["writes"]


def test_predict_files_refuses_to_run_inside_a_load_or_save():
    def nested(item):
        predict_files(SeededConv(), [0], load_func=scene, save_func=lambda *a: None, **KW)

    with pytest.raises(RuntimeError, match="inside a load_func or save_func"):
        predict_files(SeededConv(), [0], load_func=nested, save_func=lambda *a: None, **KW)


def test_a_failing_save_or_load_lets_the_next_run_through():
    def bad_save(*args):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        predict_files(SeededConv(), [0, 1], load_func=scene, save_func=bad_save, **KW)
    with IO_LOCK.write():
        pass


def _masked_array(bands, size=80):
    rng = np.random.default_rng(0)
    valid = np.ones((size, size), dtype=bool)
    valid[:10] = False
    return rng.random((bands, size, size), dtype=np.float32), valid


def test_writes_stay_on_one_thread_and_print_no_gdal_errors(tmp_path, capfd):
    """Compressing on several threads printed 'Bad value for ExtraSamples' on most small multi-band
    files, and can lose blocks with no error when it runs out of memory."""
    pytest.importorskip("rasterio")
    from manypred.geo import export_profile, write_geotiff

    array, valid = _masked_array(4)
    assert "num_threads" not in export_profile({}, array)
    for i in range(10):
        write_geotiff(tmp_path / f"{i}.tif", array, valid=valid)
    assert "ERROR" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "dtype,compress,predictor",
    [(np.float32, "deflate", 3), (np.uint8, "lzw", 1), (np.int16, "lzw", 1)],
)
def test_compression_follows_the_dtype(tmp_path, dtype, compress, predictor):
    rasterio = pytest.importorskip("rasterio")
    from manypred.geo import write_geotiff

    array, valid = _masked_array(4)
    array = (array * 100).astype(dtype)
    path = write_geotiff(tmp_path / "a.tif", array, valid=valid)
    with rasterio.open(path) as ds:
        assert ds.compression.name == compress
        assert ds.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR", "1") == str(predictor)
        assert np.array_equal(ds.read(), array)
        assert np.array_equal(ds.read_masks(1), (valid * 255).astype(np.uint8))


def test_a_float_profile_does_not_force_its_predictor_on_classes(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from manypred.geo import export_profile, write_geotiff

    floats = export_profile({}, np.zeros((1, 8, 8), dtype=np.float32))
    classes = np.ones((1, 8, 8), dtype=np.uint8)
    path = write_geotiff(tmp_path / "a.tif", classes, floats)
    with rasterio.open(path) as ds:
        assert ds.compression.name == "lzw"
        assert ds.read().min() == 1


def test_geo_helpers_take_turns_from_any_thread(tmp_path):
    pytest.importorskip("rasterio")
    from affine import Affine

    from manypred.geo import RasterioReader, load_raster, stack_rasters, write_geotiff

    profile = {"transform": Affine(1, 0, 0, 0, -1, 8), "crs": "EPSG:32750"}
    path = write_geotiff(tmp_path / "a.tif", np.ones((1, 8, 8), dtype=np.uint8), profile)
    entered = threading.Event()

    def attempt(call):
        def target():
            call()
            entered.set()

        entered.clear()
        with IO_LOCK.write():
            thread = run_thread(target)
            assert not entered.wait(0.05)  # waits for the save to finish
        thread.join(WAIT)
        assert entered.is_set()

    attempt(lambda: load_raster(path))
    attempt(lambda: stack_rasters([path]))
    attempt(lambda: RasterioReader(path).read((0, 2, 0, 2)))
    attempt(lambda: write_geotiff(tmp_path / "b.tif", np.ones((1, 8, 8), dtype=np.uint8)))
    with IO_LOCK.read(), pytest.raises(RuntimeError, match="take turns"):
        write_geotiff(tmp_path / "c.tif", np.ones((1, 8, 8), dtype=np.uint8))
