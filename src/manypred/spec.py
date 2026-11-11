"""The ``tiling_spec()`` protocol: a model describes its inputs and outputs so ManyPred fills
in scales, output resolutions, patch-size rounding and normalisation itself.

A model (e.g. a ManySeg model) may define ``tiling_spec() -> TilingSpec``; ManyPred calls it
duck-typed and never imports the model's library. It can also be passed as ``spec=``::

    {
      "inputs":  {"s2": {"kind": "raster", "scale": 1.0}, "s1": {"kind": "raster", "scale": 0.5},
                  "s3": {"kind": "raster", "scale": None},   # None: ratio from the shapes
                  "meta": {"kind": "vector"}, "ts": {"kind": "timeseries", "scale": 1.0}},
      "outputs": {"landcover": {"scope": "pixel", "scale": 1.0, "kind": "segmentation",
                                "classes": ["water", "land"]},   # optional, one name per channel
                  "flooded":   {"scope": "image", "kind": "classification"}},
      "pad_multiple": 32,     # the model pads inputs to a multiple of this
      "max_size": None,       # largest input a fixed-size backbone takes (int or (h, w))
      "normalize": None,      # optional stored normalisation spec (manypred.normalize)
      "bands": None,          # optional input band names: [...], or {input_name: [...]}
      "version": 1,           # optional: the protocol version (absent means 1)
    }

New fields come with a new ``version``; a ManyPred that doesn't know a spec's version says so
instead of misreading it. Within a version, unknown keys are errors (at the top level and in
each input and output entry), so typos fail loudly.

Explicit arguments win: ``scales=`` over input scales, ``normalize=`` over the stored spec.

Inputs declared as ``"vector"`` reach the model in float32 whatever the inference dtype (as
``<name>_dates`` do), so 16-bit inference doesn't round metadata; the model casts them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal, TypedDict

from .devices import warn_caller
from .inputs import ratio
from .normalize import validate_spec


class InputEntry(TypedDict, total=False):
    kind: Literal["raster", "timeseries", "vector"]
    scale: float | None  # None: the ratio comes from the shapes


class OutputEntry(TypedDict, total=False):
    scope: Literal["pixel", "image"]
    scale: float
    kind: str
    activation: Literal["softmax", "sigmoid", "none"]
    classes: list[str]


class TilingSpec(TypedDict, total=False):
    inputs: dict[str, InputEntry]
    outputs: dict[str, OutputEntry]
    pad_multiple: int
    max_size: int | tuple[int, int] | None
    normalize: Any
    bands: list[str] | dict[str, list[str]] | None
    version: int


SPEC_VERSION = 1  # the newest tiling spec version this ManyPred reads
_KEYS = {"inputs", "outputs", "pad_multiple", "max_size", "normalize", "bands", "version"}
_ENTRY_KEYS = {
    "inputs": set(InputEntry.__annotations__),
    "outputs": set(OutputEntry.__annotations__),
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: Any, what: str) -> Fraction:
    try:
        fraction = ratio(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{what} must be a number, got {value!r}.") from None
    if fraction <= 0:
        raise ValueError(f"{what} must be positive, got {value!r}.")
    return fraction


def _check_names(names: Any, what: str) -> None:
    if (
        isinstance(names, (str, bytes))
        or not isinstance(names, Sequence)
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError(f"{what} must be a list of names, got {names!r}.")


def validate_tiling_spec(spec: Mapping[str, Any]) -> TilingSpec:
    """Check a tiling spec; returns it (as a dict) or raises ValueError saying what is wrong."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"A tiling spec must be a dict, got {type(spec).__name__}.")
    version = spec.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"Tiling spec version must be a positive integer, got {version!r}.")
    if version > SPEC_VERSION:
        raise ValueError(
            f"This model's tiling spec is version {version}; this ManyPred reads up to "
            f"{SPEC_VERSION}. Upgrade ManyPred, or pass spec= yourself."
        )
    extra = set(spec) - _KEYS
    if extra:
        raise ValueError(
            f"Tiling spec has unexpected keys {sorted(extra)}; allowed {sorted(_KEYS)}."
        )
    for section in ("inputs", "outputs"):
        entries = spec.get(section)
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise ValueError(f"Tiling spec {section!r} must be a dict of name -> dict.")
        for name, entry in entries.items():
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Tiling spec {section} entry {name!r} must be a dict, got {entry!r}."
                )
            extra = set(entry) - _ENTRY_KEYS[section]
            if extra:
                raise ValueError(
                    f"Tiling spec {section} entry {name!r} has unexpected keys {sorted(extra)}; "
                    f"allowed {sorted(_ENTRY_KEYS[section])}."
                )
            if not isinstance(entry.get("kind", ""), (str, type(None))):
                raise ValueError(f"{section} entry {name!r}: kind must be a string.")
    for name, entry in dict(spec.get("inputs") or {}).items():
        kind = entry.get("kind", "raster")
        if kind not in ("raster", "timeseries", "vector"):
            raise ValueError(
                f"Input {name!r}: kind must be raster, timeseries or vector, got {kind!r}."
            )
        if kind != "vector" and entry.get("scale", 1.0) is not None:
            _positive(entry.get("scale", 1.0), f"Input {name!r} scale")
    for name, entry in dict(spec.get("outputs") or {}).items():
        if entry.get("activation", "none") not in ("softmax", "sigmoid", "none"):
            raise ValueError(
                f"Output {name!r}: activation must be 'softmax', 'sigmoid' or 'none', got "
                f"{entry['activation']!r}."
            )
        if entry.get("classes") is not None:
            _check_names(entry["classes"], f"Output {name!r} classes")
        scope = entry.get("scope", "pixel")
        if scope not in ("pixel", "image"):
            raise ValueError(f"Output {name!r}: scope must be 'pixel' or 'image', got {scope!r}.")
        if scope == "image":
            if "scale" in entry and entry["scale"] != 1:
                raise ValueError(f"Output {name!r} is image-level; it has no scale.")
            continue
        scale = _positive(entry.get("scale", 1.0), f"Output {name!r} scale")
        if scale.numerator != 1:
            raise ValueError(
                f"Output {name!r}: scale must be 1 / f for an integer f, got {entry['scale']!r}."
            )
    pad = spec.get("pad_multiple", 1)
    if not _is_int(pad) or pad < 1:
        raise ValueError(f"pad_multiple must be a positive integer, got {pad!r}.")
    max_size = spec.get("max_size")
    if max_size is not None:
        sizes = max_size if isinstance(max_size, Sequence) else (max_size, max_size)
        if len(sizes) != 2 or any(not _is_int(s) or s < 1 for s in sizes):
            raise ValueError(f"max_size must be None, an int or (h, w), got {max_size!r}.")
    bands = spec.get("bands")
    if bands is not None:
        for name, names in bands.items() if isinstance(bands, Mapping) else [("", bands)]:
            _check_names(names, f"Input {name!r} bands" if name else "Tiling spec bands")
    normalize = spec.get("normalize")
    if normalize is not None:
        try:
            per_input = isinstance(normalize, Mapping) and "type" not in normalize
            for steps in normalize.values() if per_input else [normalize]:
                validate_spec(steps)
        except ValueError as e:
            raise ValueError(f"Tiling spec normalisation: {e}") from None
    return dict(spec)  # type: ignore[return-value]


def get_tiling_spec(
    models: Sequence[Any], spec: Mapping[str, Any] | None, normalize: Any = None
) -> TilingSpec | None:
    """``spec`` if given, else the models' ``tiling_spec()`` (which must agree across an
    ensemble), else None. A ``normalize`` argument replaces what the models stored, so a stored
    normalisation this ManyPred can't read doesn't stop the caller supplying a good one."""
    if spec is not None:
        return validate_tiling_spec(spec)
    found = [m.tiling_spec() for m in models if callable(getattr(m, "tiling_spec", None))]
    if not found:
        return None
    if normalize is not None:
        found = [{**f, "normalize": None} if isinstance(f, Mapping) else f for f in found]
    if len(found) != len(models):
        raise ValueError(
            f"{len(found)} of the {len(models)} models in the ensemble have a tiling_spec(); "
            f"pass spec= explicitly."
        )
    if any(f != found[0] for f in found[1:]):
        raise ValueError(
            "Models in an ensemble give different tiling_spec(); pass spec= explicitly."
        )
    return validate_tiling_spec(found[0])


def output_factors(spec: TilingSpec | None) -> list[int]:
    """1 / scale of each pixel output (the multiples windows must start on)."""
    if not spec:
        return []
    return [
        int(1 / ratio(entry.get("scale", 1.0)))
        for entry in (spec.get("outputs") or {}).values()
        if entry.get("scope", "pixel") == "pixel"
    ]


def named_output_factors(spec: TilingSpec | None) -> dict[str, int]:
    """``{output: 1 / scale}`` of the pixel outputs whose scale a spec gives."""
    outputs = (spec or {}).get("outputs") or {}
    return {
        name: int(1 / ratio(entry["scale"]))
        for name, entry in outputs.items()
        if entry.get("scope", "pixel") == "pixel" and entry.get("scale") is not None
    }


def output_activations(spec: TilingSpec | None) -> dict[str, str]:
    """``{output: activation}`` the model declares, to use when none is given."""
    if not spec:
        return {}
    return {
        name: entry["activation"]
        for name, entry in (spec.get("outputs") or {}).items()
        if entry.get("activation") is not None
    }


def output_classes(spec: TilingSpec | None) -> dict[str, list[str]]:
    """``{output: class names}`` the model declares."""
    return {
        name: list(entry["classes"])
        for name, entry in ((spec or {}).get("outputs") or {}).items()
        if entry.get("classes") is not None
    }


def regression_outputs(spec: TilingSpec | None) -> list[str]:
    """The outputs the model declares as regression: their channels aren't classes."""
    return [
        name
        for name, entry in ((spec or {}).get("outputs") or {}).items()
        if entry.get("kind") == "regression"
    ]


def output_names(spec: TilingSpec | None) -> list[str]:
    """The model's output names, as its spec lists them."""
    return list((spec or {}).get("outputs") or {})


def output_alias(spec: TilingSpec | None) -> str | None:
    """The name a spec gives a model's only output, which settings may use for a head the
    model returns as a bare tensor; None unless the spec lists exactly one output."""
    names = output_names(spec)
    return names[0] if len(names) == 1 else None


def vector_inputs(spec: TilingSpec | None) -> frozenset[str]:
    """The inputs a spec declares as vectors: they reach the model in float32 whatever the
    inference dtype (the model casts them), so 16-bit inference doesn't round metadata."""
    inputs = (spec or {}).get("inputs") or {}
    return frozenset(name for name, entry in inputs.items() if entry.get("kind") == "vector")


def input_order(spec: TilingSpec | None) -> list[str]:
    """The model's input names in its spec's order: the first raster is its primary input."""
    return list((spec or {}).get("inputs") or {})


def input_scales(spec: TilingSpec | None) -> dict[str, float]:
    if not spec:
        return {}
    return {
        name: entry.get("scale", 1.0)
        for name, entry in (spec.get("inputs") or {}).items()
        if entry.get("kind", "raster") != "vector" and entry.get("scale", 1.0) is not None
    }


def fit_patch(spec: TilingSpec | None, patch_size: int, overlap: int) -> tuple[int, int]:
    """Cap the patch at ``max_size`` (with a warning, and the overlap at half of it: an overlap
    meant for a larger patch could leave a stride of one pixel) and round it down to
    ``pad_multiple`` (silently: the model would pad it up anyway)."""
    if not spec:
        return patch_size, overlap
    max_size = spec.get("max_size")
    if max_size is not None:
        limit = min(max_size) if isinstance(max_size, Sequence) else max_size
        if patch_size > limit:
            warn_caller(
                f"patch_size {patch_size} is larger than the model's max_size {max_size}; "
                f"using {limit}."
            )
            patch_size = limit
            overlap = min(overlap, limit // 2)
    pad = spec.get("pad_multiple", 1)
    if patch_size >= pad:
        patch_size -= patch_size % pad
    overlap = min(overlap, patch_size - 1)
    return patch_size, overlap
