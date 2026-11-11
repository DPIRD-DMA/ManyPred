# Notebooks

Longer walkthroughs with real results and figures. Each one is saved with its outputs, so you can
read it here without running anything. For short reference code on one feature, use the
[scripts in `examples/`](../examples/).

All four run on real Sentinel-2 tiles from the Kimberley in Western Australia. The saved timings are
from an RTX 4090 that another process was also using.

| Notebook | Shows | Downloads | To run it |
| --- | --- | --- | --- |
| [Cloud-masking a whole tile](01_cloud_mask_whole_tile.ipynb) | OmniCloudMask's released models over 10980 x 10980 px: patch planning, what overlap does to seams, nodata handling, agreement with OmniCloudMask itself, GeoTIFF output | About 500 MB, plus the weights | A couple of minutes on a GPU |
| [From a trained model to a flood map](02_flood_scene.ipynb) | A ManySeg model trained, saved with its settings and run over the Fitzroy River flood of January 2023, against the same tile in the dry season | About 1.3 GB | A couple of minutes on a GPU |
| [Two resolutions in, three kinds of output out](03_two_grids_many_heads.ipynb) | 10 m and 20 m inputs with a full-resolution head, a 40 m head and an image-level head, stitched in one call and written as one GeoTIFF per head | About 0.9 GB, nothing after notebook 2 | A couple of minutes on a GPU |
| [From a tile to polygons, one block at a time](04_blocks_to_polygons.ipynb) | A lazy read, `merge="cut"`, post-processing the saved raster by block, retuning without the model, and GeoJSON written during prediction | About 500 MB, nothing after notebook 1 | A few minutes on a GPU |

## Running them

```bash
pip install "manypred[geo] @ git+https://github.com/DPIRD-DMA/ManyPred" \
    "manyseg @ git+https://github.com/DPIRD-DMA/ManySeg#subdirectory=packages/manyseg" \
    omnicloudmask matplotlib scipy shapely jupyter
jupyter lab
```

Downloads go to `data/` next to the notebook, and trained models to `runs/`. Set `MANYPRED_DATA` to
put the downloads somewhere else. Notebook 4 also writes about 2 GB of rasters there.

## Results

The training in notebooks 2 and 3 is short and uses one seed, on chips from other floods. Their
scores and flood areas show the workflow, and they don't measure what these models can do. For
OmniCloudMask's accuracy see [its repository](https://github.com/DPIRD-DMA/OmniCloudMask).

## Tests

`tests/test_notebooks.py` checks on every commit that the saved notebooks have their figures, stay
small and contain no local paths. The weekly slow tier runs each one from scratch with
`MANYPRED_NOTEBOOK_SMOKE=1`, which uses 160 m pixels, a few chips and one epoch. That shows the code
still runs, but not that the saved results are still current, so re-run a notebook after changing
what it demonstrates. To run the smoke test locally:

```bash
uv run --no-sync --with nbclient --with nbformat --with ipykernel --with matplotlib \
  --with scipy --with shapely python -m pytest tests/test_notebooks.py -m "slow and network"
```

Each notebook skips if a package it needs is missing. CI's slow tier has `omnicloudmask` but not
`manyseg`, so it runs notebooks 1 and 4 only.
