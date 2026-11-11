"""The notebooks in ``notebooks/``: saved outputs on every commit, a run from scratch in the slow
tier.

The run sets ``MANYPRED_NOTEBOOK_SMOKE=1`` (160 m pixels, a few chips, one epoch, no pretrained
weights) and works on a copy, so the saved outputs stay as they are. It reads Sentinel-2 overviews
and a few Sen1Floods11 chips over the network, and skips if a bucket can't be reached. The sdist
has no notebooks, so these tests collect nothing there.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
FILES = sorted(NOTEBOOKS.glob("*.ipynb"))
IDS = [p.stem for p in FILES]
BUCKETS = (
    "https://storage.googleapis.com/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson",
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/51/K/YV/2023/1/"
    "S2B_51KYV_20230112_0_L2A/B03.tif",
)
MAX_MB = 3  # saved figures are committed: keep their resolution down
LOCAL_PATH = re.compile(r"/home/|/media/|/Users/|C:\\\\Users")


def _cells(path):
    return json.loads(path.read_text())["cells"]


def _code(path):
    return [c for c in _cells(path) if c["cell_type"] == "code"]


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_notebook_is_saved_with_outputs(path):
    code = _code(path)
    images = sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"])
    assert images >= 3, f"{path.name} has {images} saved figures"
    assert not [o for c in code for o in c["outputs"] if o["output_type"] == "error"], path.name


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_notebook_is_small_and_has_no_local_paths(path):
    assert path.stat().st_size < MAX_MB * 1e6, f"{path.name} is over {MAX_MB} MB"
    text = "\n".join("".join(o.get("text", [])) for c in _code(path) for o in c["outputs"])
    assert not LOCAL_PATH.search(text), f"{path.name} prints a local path"
    assert not LOCAL_PATH.search("\n".join("".join(c["source"]) for c in _cells(path))), path.name


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_notebook_runs(path, tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    source = path.read_text()
    if "omnicloudmask" in source:
        pytest.importorskip("omnicloudmask")
    if "manyseg" in source:
        pytest.importorskip("manyseg")
    for url in BUCKETS:
        try:
            urllib.request.urlopen(url, timeout=20).close()
        except urllib.error.URLError as e:
            pytest.skip(f"{url} not reachable: {e}")
    for key, value in {
        "MANYPRED_NOTEBOOK_SMOKE": "1",
        "MANYPRED_DATA": str(tmp_path / "data"),
        "CUDA_VISIBLE_DEVICES": "",
        "MPLBACKEND": "agg",
        "PYDEVD_DISABLE_FILE_VALIDATION": "1",
        "OMP_NUM_THREADS": os.environ.get("MANYPRED_TEST_THREADS", "4"),
    }.items():
        monkeypatch.setenv(key, value)
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
