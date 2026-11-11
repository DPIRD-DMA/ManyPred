"""A few seeds of scripts/fuzz_manyseg.py (slow tier; the full fuzzer runs from the script)."""

import random
import sys
from pathlib import Path

import pytest

SEEDS = [0, 3, 7, 20, 31, 48]


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_seed(seed, tmp_path):
    manyseg = pytest.importorskip("manyseg")
    if not hasattr(manyseg, "save"):
        pytest.skip("needs a ManySeg with save/load")
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import fuzz_manyseg

    findings = fuzz_manyseg.run_checks(seed, fuzz_manyseg.sample(random.Random(seed)), tmp_path)
    bugs = [f for f in findings if f["check"] != "rejected"]
    assert not bugs, bugs
