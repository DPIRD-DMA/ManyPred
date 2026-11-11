"""The golden files are intact and, with OmniCloudMask installed (``parity`` group), still
what OCM at the pinned commit produces."""

import hashlib
import json
import sys
from pathlib import Path

import goldens
import numpy as np
import pytest
import torch

MANIFEST = json.loads((goldens.GOLDEN_DIR / "manifest.json").read_text())


def test_golden_files_match_manifest():
    for rel, digest in MANIFEST["sha256"].items():
        data = (goldens.GOLDEN_DIR / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
    assert set(MANIFEST["cases"]) == set(goldens.case_names())


def test_golden_files_fit_budget():
    assert MANIFEST["total_bytes"] < 20e6


def test_case_inputs_rebuild():
    inputs = goldens.load_inputs()
    for name in goldens.case_names():
        case = goldens.load_case(name)
        meta = case["meta"]
        if meta["func"] == "predict_from_array":
            image = goldens.make_image(inputs, meta["input"])
            assert case["output"].shape[1:] == image.shape[1:], name
        elif meta["func"] == "predict_from_batch":
            chips = goldens.make_chips(inputs, meta["input"])
            assert case["output"].shape[0] == len(chips), name


@pytest.mark.parametrize("name", goldens.TINY_MODELS)
def test_tiny_models_load(name):
    model = goldens.load_tiny(name)
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 37, 41))
    assert out.shape == (2, 4, 37, 41)


def test_goldens_reproduce_with_ocm():
    """Rerun the tiny-model cases with OCM live and compare: classes exactly, values within
    1e-6 where OCM is deterministic (see scripts/make_goldens.py), else 1e-5."""
    pytest.importorskip("omnicloudmask")
    pytest.importorskip("rasterio")
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import make_goldens

    make_goldens.check_ocm_commit()
    inputs = goldens.load_inputs()
    cases = make_goldens.build_cases(include_ocm=False)
    for name, case in cases.items():
        new = make_goldens.run_case(case, inputs, make_goldens.DEFAULT_WEIGHTS)
        old = goldens.load_case(name)
        exact = MANIFEST["cases"][name]["deterministic"]
        for key, value in new.items():
            if not isinstance(value, np.ndarray) or value.dtype.kind in "US":
                continue
            if value.dtype.kind != "f":
                np.testing.assert_array_equal(value, old[key], err_msg=f"{name}:{key}")
            else:
                # deterministic cases are bitwise only on the CPU that generated them; other
                # CPUs' kernels differ by up to ~7e-7 (seen on CI runners)
                atol = 1e-6 if exact else 1e-5
                np.testing.assert_allclose(value, old[key], atol=atol, err_msg=f"{name}:{key}")
