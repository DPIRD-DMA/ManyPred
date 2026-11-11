# API reference

## Prediction

```{eval-rst}
.. autofunction:: manypred.predict_array
.. autofunction:: manypred.predict_batch
.. autofunction:: manypred.predict_files
.. autoclass:: manypred.TilePredictions
   :members: to_map, reduce
```

## Inputs

```{eval-rst}
.. automodule:: manypred.inputs
   :no-members:
```

## Outputs

```{eval-rst}
.. automodule:: manypred.outputs
   :no-members:
```

## Block post-processing

```{eval-rst}
.. automodule:: manypred.blocks
   :no-members:
.. autofunction:: manypred.postprocess_blocks
.. autoclass:: manypred.BlockPostprocess
   :no-members:
.. autofunction:: manypred.blocks.block_grid
.. autoclass:: manypred.blocks.Block
   :members: inner
.. autoclass:: manypred.blocks.Scene
```

## Normalisation

```{eval-rst}
.. automodule:: manypred.normalize
   :no-members:
.. autofunction:: manypred.normalize_numpy
.. autofunction:: manypred.normalize_torch
.. autofunction:: manypred.normalize.validate_spec
```

## The `tiling_spec()` protocol

```{eval-rst}
.. automodule:: manypred.spec
   :no-members:
.. autoclass:: manypred.spec.TilingSpec
.. autoclass:: manypred.spec.InputEntry
.. autoclass:: manypred.spec.OutputEntry
.. autofunction:: manypred.spec.validate_tiling_spec
```

## Rasters and GeoTIFFs (`[geo]`)

```{eval-rst}
.. automodule:: manypred.geo
   :no-members:
.. autofunction:: manypred.geo.load_raster
.. autofunction:: manypred.geo.stack_rasters
.. autoclass:: manypred.geo.RasterioReader
.. autoclass:: manypred.geo.GeoTIFFWriter
   :no-members:
.. autofunction:: manypred.geo.write_geotiff
.. autofunction:: manypred.geo.export_profile
.. autofunction:: manypred.geo.default_output_path
.. automodule:: manypred.io_lock
   :no-members:
```

## Lazy readers

```{eval-rst}
.. automodule:: manypred.lazy
   :no-members:
.. autoclass:: manypred.lazy.Reader
```

## Blend weights

```{eval-rst}
.. autofunction:: manypred.blend.gradient_weights
```
