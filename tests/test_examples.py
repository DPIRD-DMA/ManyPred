"""The examples run (slow tier: they start fresh processes)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.stem for e in EXAMPLES])
def test_example_runs(example, tmp_path):
    if example.stem in ("02_files", "04_large_scene", "07_blocks"):
        pytest.importorskip("rasterio")
    if example.stem.startswith("06_manyseg"):
        manyseg = pytest.importorskip("manyseg")
        if not hasattr(manyseg, "save"):
            pytest.skip("needs a ManySeg with save/load")
    result = subprocess.run(
        [sys.executable, str(example), str(tmp_path)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
