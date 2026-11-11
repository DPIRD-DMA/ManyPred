# Changelog

All notable changes are listed here. Until 1.0, a minor version (0.x) may change the API; each
change is listed under "Changed".

## [0.1.0] - 2026-11-11

First release. ManyPred is the inference engine of
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (at commit `50b8f5f`) as its own
library, generalised to any PyTorch model that returns a tensor or a dict of tensors.

### Added

- `predict_array`: tiled inference over a scene. Overlapping patches on a grid with clamped
  edges, all-nodata patches skipped, patches shifted off nodata edges (`shift_patches`) and
  optionally shrunk over large nodata areas (`adapt_patch_size`), blended with OmniCloudMask's
  gradient weights in float32, float16 or bfloat16 (`mosaic_dtype`), on the inference device or
  the CPU (`mosaic_device`). `merge="cut"` takes each pixel from its most central patch instead
  of blending.
- `predict_batch`: many same-sized chips, several per batch, with no blending;
  `group_by_shape=True` for mixed sizes.
- `predict_files`: files in, files out. Loading, inference and saving overlap on background
  threads. `overwrite=False` skips finished items. The default GeoTIFF writer (`[geo]` extra)
  gives files their final names only when complete, and refuses before any work a run in which
  two items share an output or an output is also an input.
- Outputs: `"logits"`, `"probs"`, `"argmax"` or a callable, with per-output settings for dict
  outputs; reduced-resolution heads blended on their own grid; image-level heads as
  `TilePredictions`; `postprocess` on each finished prediction. Missing pixels are NaN in float
  outputs and 0 in class outputs.
- Inputs: arrays or tensors, dicts of rasters at other resolutions (`scales`), time series and
  vectors, and lazy readers for scenes larger than memory, with outputs written in row blocks.
- Normalisation specs: `scale`, `zscore` (dataset, tile or scene), `minmax`, `percentile`, `db`,
  `clip` and callables, per input; `normalize_numpy` and `normalize_torch`, and `validate_spec`
  to check a normalisation spec.
- The `tiling_spec()` protocol (version 1): models describe their inputs, outputs, padding, size
  limits, normalisation, activations, class names and bands, and ManyPred uses them. ManySeg
  models implement it. `manypred.spec.validate_tiling_spec` checks a tiling spec.
- Ensembles (raw outputs averaged), float16 and bfloat16 inference, `torch.compile` (short
  batches padded to the full size), CUDA, MPS and CPU, with pinned-memory copies overlapping
  compute on CUDA. Callables may declare `batch_size`, `fixed_size` and `device`, for exported
  programs.
- `manypred.geo`: `load_raster` and `stack_rasters` (resampling that keeps the extent;
  `grid="source"` writes predictions back at the file's resolution), `RasterioReader`,
  `write_geotiff` and `GeoTIFFWriter`. Loads and saves through GDAL take turns across threads.
- `manypred.blocks`: `postprocess_blocks` runs a function on each block (core plus halo) of a
  fixed grid over stitched rasters, arrays or raster files, optionally in worker processes;
  `predict_files(block_postprocess=...)` does the same while predicting.
- Parity with OmniCloudMask, checked against golden files: batch outputs within 1e-5, tiled
  logits within 1e-4 and classes equal except at near-ties.
- `notebooks/`: walkthroughs on real Sentinel-2 tiles.
