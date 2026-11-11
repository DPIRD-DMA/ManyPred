# ManyPred

Tiled and batched inference for PyTorch dense-prediction models. Pass an array and get the
prediction back, or pass a list of files and get predictions written.

```{image} assets/pipeline-light.svg
:alt: "ManyPred pipeline: a scene is cut into overlapping patches, nodata patches are skipped, batches run through normalisation and the model on the GPU, predictions are blended with weights that fade at patch edges, and the outputs are mosaics, reduced-resolution heads, image-level values or GeoTIFFs"
:align: center
```

```{include} ../README.md
:start-after: "<!-- docs-start -->"
```

```{toctree}
:hidden:
:maxdepth: 2

api
changelog
```
