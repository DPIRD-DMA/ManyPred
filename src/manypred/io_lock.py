"""Loads and saves take turns, across the process.

A GeoTIFF with an internal mask or overviews does unlocked file I/O when it closes, while the
mask's dirty blocks are still in GDAL's process-wide block cache. Any other thread that flushes
that cache then writes them at the same time, which can segfault, hang, or write a bad mask with
at most a printed TIFF error (OmniCloudMask#52). Triggers include decoding a JPEG2000 file,
writing another GeoTIFF once the cache is full, and changing GDAL_CACHEMAX.

So :func:`~manypred.predict_files` and ManyPred's GDAL helpers hold the read side while loading
and reading and the write side while saving. Inference overlaps both. GDAL calls made anywhere
else in the process (``postprocess``, a block ``fn``, other libraries) aren't covered."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition, local


class ReadWriteLock:
    """Any number of readers at once, or one writer. A waiting writer holds back new readers, so
    a stream of reads can't starve it. Both sides are re-entrant, and the writer may read."""

    def __init__(self) -> None:
        self._reset()
        if hasattr(os, "register_at_fork"):
            # a forked child gets the counts but not the threads that would release them
            os.register_at_fork(after_in_child=self._reset)

    def _reset(self) -> None:
        self._condition = Condition()
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0
        self._local = local()

    def held(self) -> bool:
        """Whether this thread holds either side."""
        return bool(getattr(self._local, "reads", 0) or getattr(self._local, "writes", 0))

    @contextmanager
    def read(self) -> Iterator[None]:
        reads = getattr(self._local, "reads", 0)
        if reads or getattr(self._local, "writes", 0):
            self._local.reads = reads + 1
            try:
                yield
            finally:
                self._local.reads = reads
            return
        with self._condition:
            while self._writing or self._writers_waiting:
                self._condition.wait()
            self._readers += 1
        self._local.reads = 1
        try:
            yield
        finally:
            self._local.reads = 0
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        if getattr(self._local, "reads", 0) and not getattr(self._local, "writes", 0):
            # it would wait forever for this thread's own read to end
            raise RuntimeError(
                "Can't save while loading, e.g. from inside a load_func: loads and saves take "
                "turns."
            )
        writes = getattr(self._local, "writes", 0)
        if writes:
            self._local.writes = writes + 1
            try:
                yield
            finally:
                self._local.writes = writes
            return
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writing or self._readers:
                    self._condition.wait()
            finally:
                self._writers_waiting -= 1
                # wake readers held back by this writer if the wait was interrupted
                self._condition.notify_all()
            self._writing = True
        self._local.writes = 1
        try:
            yield
        finally:
            self._local.writes = 0
            with self._condition:
                self._writing = False
                self._condition.notify_all()


IO_LOCK = ReadWriteLock()
