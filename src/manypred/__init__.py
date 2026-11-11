"""ManyPred: tiled and batched inference for PyTorch dense-prediction models."""

from importlib.metadata import PackageNotFoundError, version

from .batch import predict_batch
from .blocks import BlockPostprocess, postprocess_blocks
from .files import predict_files
from .normalize import normalize_numpy, normalize_torch, validate_spec
from .outputs import TilePredictions
from .spec import TilingSpec
from .tiled import predict_array

try:
    __version__ = version("manypred")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "BlockPostprocess",
    "TilePredictions",
    "TilingSpec",
    "__version__",
    "normalize_numpy",
    "normalize_torch",
    "postprocess_blocks",
    "predict_array",
    "predict_batch",
    "predict_files",
    "validate_spec",
]
