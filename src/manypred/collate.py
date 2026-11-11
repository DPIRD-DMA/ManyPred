"""Collating images into batches ready to copy to the device, keeping small dtypes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .devices import same_device

# Raster dtypes sent to the device as they are and cast to float32 there: small ones, and
# those float32 can't hold exactly, so nodata is compared with the original values. Others
# are cast on the host (see _raster_transfer_dtype).
_TRANSFER_DTYPES = {
    np.dtype(dtype)
    for dtype in (
        np.uint8,
        np.int8,
        np.int16,
        np.uint16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    )
}


def _fits_int64(images: Any) -> bool:
    """Whether every value of these arrays (or one array) fits int64."""
    parts = [images] if isinstance(images, np.ndarray) else list(images or [])
    return all(
        part.size == 0 or part.dtype != np.uint64 or int(part.max()) <= np.iinfo(np.int64).max
        for part in parts
    )


def _common_dtype(dtypes: set[np.dtype], images: Any) -> np.dtype:
    """One dtype for images of several: numpy's promotion, which holds every value, so a
    chip's values (and its nodata comparison) don't depend on what it is batched with: int32
    and int64 stay integers, uint8 with float16 is float16, int32 with float32 is float64. The
    exception is int64 with uint64, which goes as int64 when every value fits (numpy would
    widen it to float64). MPS has no float64; it gets float32 later, as for one dtype."""
    if len(dtypes) == 1:
        return next(iter(dtypes))
    dtype = np.result_type(*dtypes)
    if dtype.kind == "f" and all(d.kind in "biu" for d in dtypes):
        return np.dtype(np.int64) if _fits_int64(images) else np.dtype(np.float64)
    return dtype


def _raster_transfer_dtype(dtype: np.dtype, device: torch.device, images: Any = None) -> np.dtype:
    """The dtype a raster batch travels in. uint32 goes as int64 (torch supports few uint32
    operations), and so does uint64 when every value of ``images`` fits (so nodata is compared
    exactly), else as float64; float64 goes as float32 on MPS, which has no float64; anything
    else not in ``_TRANSFER_DTYPES`` (bool, ...) as float32."""
    if dtype == np.uint32:
        return np.dtype(np.int64)
    if dtype == np.uint64:
        if _fits_int64(images):
            return np.dtype(np.int64)
        dtype = np.dtype(np.float64)
    if dtype == np.float64 and device.type == "mps":
        return np.dtype(np.float32)
    return dtype if dtype in _TRANSFER_DTYPES else np.dtype(np.float32)


# key for the input of single-input models in the internal {name: tensor} dicts
SINGLE = ""


def _host_dtype(image: Any) -> np.dtype | None:
    """The numpy dtype a host image is collated from (bfloat16 as float32, as
    :func:`to_host_array` gives it); None for a tensor on another device."""
    if isinstance(image, np.ndarray):
        return image.dtype
    if isinstance(image, torch.Tensor):
        if image.device.type != "cpu":
            return None
        if image.dtype == torch.bfloat16:
            return np.dtype(np.float32)
        return torch.empty(0, dtype=image.dtype).numpy().dtype
    return np.asarray(image).dtype


def chip_dtypes(images: Any) -> list[np.dtype] | None:
    """Each chip's dtype, for a batch of rasters that mixes a float dtype with others (it
    travels in one dtype that holds them all, where nodata must still be compared as each
    chip's own dtype holds it: see :func:`chip_nodata`); None otherwise."""
    if isinstance(images, (np.ndarray, torch.Tensor)):
        return None
    dtypes = [_host_dtype(image) for image in images]
    if any(d is None for d in dtypes):  # not `None in dtypes`: np.dtype(None) is float64
        return None
    if len(set(dtypes)) < 2 or not any(d.kind == "f" for d in dtypes):
        return None
    if len(images[0].shape) < 3:  # vectors are never compared with nodata
        return None
    return dtypes


def chip_nodata(dtypes: Sequence[np.dtype], nodata: float) -> list[float]:
    """``nodata`` as each chip's dtype holds it, so a chip of a mixed batch matches it as it
    would batched alone: rounded for float16 and float32 chips, exact for integer chips, and
    NaN (matching nothing) where an integer dtype can't hold it."""
    values = []
    for dtype in dtypes:
        if dtype.kind == "f":
            values.append(float(np.asarray(nodata).astype(dtype)))
            continue
        # checked as an exact integer (float rounds 2**63 - 1 up past int64), then rounded to
        # a float as the chip's values are on the way to the batch's dtype
        if isinstance(nodata, (int, np.integer)) and not isinstance(nodata, bool):
            whole: int | None = int(nodata)
        else:
            number = float(nodata)
            whole = int(number) if math.isfinite(number) and number.is_integer() else None
        info = np.iinfo(dtype) if dtype.kind in "iu" else None
        fits = whole is not None and (info is None or info.min <= whole <= info.max)
        values.append(float(whole) if fits else math.nan)
    return values


def as_inputs(image: Any) -> dict[str, Any]:
    """An image as {input_name: array or tensor}; a single array or tensor is keyed
    ``SINGLE``."""
    if isinstance(image, Mapping):
        if not image:
            raise ValueError("An input dict must hold at least one input.")
        return dict(image)
    return {SINGLE: image}


def to_host_array(image: Any) -> np.ndarray | torch.Tensor:
    """numpy for host data (bfloat16 as float32, as numpy has no bfloat16); device tensors
    are returned as they are."""
    if isinstance(image, torch.Tensor):
        if image.device.type != "cpu":
            return image
        return (image.float() if image.dtype == torch.bfloat16 else image).numpy()
    return np.asarray(image)


def collate(images: Sequence[Any] | np.ndarray | torch.Tensor, device: torch.device):
    """Collect a batch of same-shape images into one contiguous tensor, ready to copy to
    ``device``. ``images`` is a sequence of arrays/tensors, or an already stacked array or
    tensor (e.g. a slice of a 4D array, collated without stacking).

    Returns ``(tensor, is_uint16)``: uint16 data is sent viewed as int16, which torch does
    not fully support, and restored by :func:`to_float32` on the device. Host images are
    copied once, in their transfer dtype (:func:`_raster_transfer_dtype`), into a buffer
    pinned for fast copies to CUDA. Tensors
    on another device are moved one by one and stacked there; tensors already on ``device``
    are used as they are."""
    if isinstance(images, torch.Tensor):
        if images.device.type != "cpu" and same_device(images.device, device):
            return images, False
        if images.device.type != "cpu":
            # moving a sliced MPS batch to the CPU in one call has returned the wrong rows,
            # so images go one by one
            images = list(images)
        else:
            images = to_host_array(images)

    if not isinstance(images, np.ndarray):
        images = list(images)
        if any(isinstance(i, torch.Tensor) and i.device.type != "cpu" for i in images):
            return torch.stack([torch.as_tensor(i).to(device=device) for i in images]), False
        images = [to_host_array(i) for i in images]
        dtypes = {image.dtype for image in images}
        shape = (len(images), *images[0].shape)
    else:
        dtypes = {images.dtype}
        shape = images.shape

    dtype = _common_dtype(dtypes, images)
    vectors = len(shape) == 2
    if vectors:
        # vectors (masks, dates) keep their dtype, skipping the int16 transport of uint16
        if dtype == np.uint16:
            dtype = np.dtype(np.int32)
        elif dtype == np.float64 and device.type == "mps":  # MPS has no float64
            dtype = np.dtype(np.float32)
        elif dtype.kind not in "biu" and dtype not in _TRANSFER_DTYPES:
            dtype = np.dtype(np.float32)
    else:
        dtype = _raster_transfer_dtype(dtype, device, images)
    is_uint16 = dtype == np.uint16
    buffer_dtype = np.dtype(np.int16) if is_uint16 else dtype

    if device.type == "cuda":
        torch_dtype = torch.from_numpy(np.empty(0, dtype=buffer_dtype)).dtype
        buffer = torch.empty(shape, dtype=torch_dtype, pin_memory=True)
    else:
        buffer = torch.from_numpy(np.empty(shape, dtype=buffer_dtype))
    # numpy assignment handles any strides and casts other dtypes; uint16 data is written
    # through a uint16 view of the int16 buffer, so other dtypes in the batch are cast, not
    # reinterpreted
    target = buffer.numpy()
    if is_uint16:
        target = target.view(np.uint16)
    if isinstance(images, np.ndarray):
        target[...] = images
    else:
        for i, image in enumerate(images):
            target[i] = image
    return buffer, is_uint16


def to_float32(batch: torch.Tensor, is_uint16: bool) -> torch.Tensor:
    """Cast a collated batch to float32, restoring uint16 values sent as int16."""
    if is_uint16:
        try:
            # one cast kernel, where the device supports uint16
            return batch.view(torch.uint16).to(torch.float32)
        except (RuntimeError, TypeError):
            return batch.to(torch.int32).bitwise_and_(0xFFFF).to(torch.float32)
    return batch.to(torch.float32)


def tensor_devices(images: Any) -> list[torch.device]:
    """Devices of the tensors in a batch of images (a tensor, or a list of them)."""
    if isinstance(images, torch.Tensor):
        return [images.device]
    if isinstance(images, (list, tuple)):
        return [i.device for i in images if isinstance(i, torch.Tensor)]
    return []


def touches_mps(images: Any) -> bool:
    """Whether collating these images would do MPS work. MPS is not thread-safe, so such
    batches are collated by the runner under the device lock, not on the prefetch thread."""
    return any(device.type == "mps" for device in tensor_devices(images))
