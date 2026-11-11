"""Programs: callables that declare a fixed batch size, input size or device, as
``manyseg.load`` does for a saved ``quantize_pt2e`` model."""

import warnings

import numpy as np
import pytest
import torch

from manypred import predict_array, predict_batch, predict_files
from manypred.runner import program_device, program_patch_size
from manypred.spec import fit_patch

KW = {"device": "cpu", "progress": False}


class Program:
    """A conv behind the checks a program makes. Records the batch shapes it was called with."""

    def __init__(self, batch_size=None, fixed_size=None, device=None, seed=0):
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, 2, 3, padding=1).eval()
        self.batch_size, self.fixed_size = batch_size, fixed_size
        if device is not None:
            self.device = torch.device(device)
        self.shapes = []

    def __call__(self, x):
        self.shapes.append(tuple(x.shape))
        if self.batch_size is not None and x.shape[0] != self.batch_size:
            raise ValueError(f"needs batch size {self.batch_size}, got {x.shape[0]}")
        if self.fixed_size is not None and tuple(x.shape[-2:]) != tuple(self.fixed_size):
            raise ValueError(f"needs size {self.fixed_size}, got {tuple(x.shape[-2:])}")
        return self.conv(x)


class SpecProgram(Program):
    """A program whose ``tiling_spec()`` disagrees with its fixed size."""

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

    def tiling_spec(self):
        return self.spec


def scene(height=96, width=112):
    return np.random.default_rng(0).random((3, height, width)).astype(np.float32)


TILED = {"patch_size": 64, "overlap": 16, **KW}


@pytest.mark.parametrize("size", [1, 2, 3, 4, 50])
def test_a_fixed_batch_gives_what_a_dynamic_one_gives(size):
    """Every call has the program's batch size, including the short last batch."""
    program = Program(batch_size=size)
    out = predict_array(program, scene(), **TILED)
    assert {shape[0] for shape in program.shapes} == {size}
    np.testing.assert_allclose(out, predict_array(Program(), scene(), **TILED), atol=1e-5)


def test_asking_for_another_batch_size_warns_and_uses_the_programs():
    program = Program(batch_size=3)
    with pytest.warns(UserWarning, match="batch size 3; using it, not 2"):
        predict_array(program, scene(), batch_size=2, **TILED)
    assert {shape[0] for shape in program.shapes} == {3}


@pytest.mark.parametrize("call", ["array", "batch"])
def test_the_default_batch_size_doesnt_warn(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if call == "array":
            predict_array(Program(batch_size=3), scene(), **TILED)
        else:
            predict_batch(Program(batch_size=3), [scene(), scene()], **KW)


def test_predict_batch_pads_a_short_batch():
    program = Program(batch_size=4)
    chips = [scene() for _ in range(6)]
    out = predict_batch(program, chips, **KW)
    assert len(out) == 6 and {shape[0] for shape in program.shapes} == {4}
    np.testing.assert_allclose(out, predict_batch(Program(), chips, **KW), atol=1e-5)


@pytest.mark.parametrize("size", [3, 4])
def test_compile_uses_the_programs_batch_size(monkeypatch, size):
    """The first batch may be shorter than the program's size: warm-up must not hit its check.
    torch.compile needs a C++ compiler, which CI's Windows has not, and isn't under test here."""
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    program = Program(batch_size=size)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="torch.compile failed")
        out = predict_array(program, scene(64, 64), compile=True, **TILED)  # one patch
    assert {shape[0] for shape in program.shapes} == {size}
    np.testing.assert_allclose(out, predict_array(Program(), scene(64, 64), **TILED), atol=1e-4)


def test_a_failed_compile_still_pads(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(torch, "compile", fail)
    program = Program(batch_size=3)
    with pytest.warns(UserWarning, match="torch.compile failed"):
        predict_array(program, scene(), compile=True, **TILED)
    assert {shape[0] for shape in program.shapes} == {3}


def test_a_square_fixed_size_sets_the_patch():
    program = Program(fixed_size=(64, 64))
    with pytest.warns(UserWarning, match="using patch_size=64 and overlap=32"):
        out = predict_array(program, scene(), patch_size=1000, overlap=300, **KW)
    assert out.shape == (2, 96, 112)
    assert {shape[-2:] for shape in program.shapes} == {(64, 64)}


def test_a_patch_smaller_than_the_fixed_size_is_raised_to_it():
    program = Program(fixed_size=(64, 64))
    with pytest.warns(UserWarning, match="using patch_size=64"):
        predict_array(program, scene(), patch_size=32, overlap=8, **KW)
    assert {shape[-2:] for shape in program.shapes} == {(64, 64)}


def test_the_right_patch_size_doesnt_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict_array(Program(fixed_size=(64, 64)), scene(), **TILED)


def test_a_non_square_fixed_size_cant_be_tiled():
    with pytest.raises(ValueError, match="fixed 64 x 96.*square"):
        predict_array(Program(fixed_size=(64, 96)), scene(), **TILED)


def test_a_non_square_fixed_size_runs_chips_whole():
    program = Program(fixed_size=(64, 96))
    assert predict_batch(program, [scene(64, 96)], **KW).shape[-2:] == (64, 96)


def test_adapt_patch_size_is_refused_at_a_fixed_size():
    with pytest.raises(ValueError, match="adapt_patch_size"):
        predict_array(Program(fixed_size=(64, 64)), scene(), adapt_patch_size=True, **TILED)


@pytest.mark.parametrize("shape", [(40, 40), (40, 112), (6, 6)])
def test_a_scene_smaller_than_the_fixed_size_is_padded_to_it(shape):
    """The program only ever gets its fixed size: the patch is padded past the scene and the
    output cropped back, without a warning."""
    program = Program(fixed_size=(64, 64))
    image = scene(*shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = predict_array(program, image, **TILED)
    assert out.shape == (2, *shape)
    assert {s[-2:] for s in program.shapes} == {(64, 64)}
    padded = np.zeros((3, 64, 64), np.float32)
    padded[:, : shape[0], : min(shape[1], 64)] = image[:, :, :64]
    expected = program.conv(torch.from_numpy(padded)[None])[0, :, : shape[0], : min(shape[1], 64)]
    np.testing.assert_allclose(
        out[:, :, : min(shape[1], 64)][:, :, :48], expected.detach().numpy()[:, :, :48], atol=1e-5
    )


def test_files_use_the_programs_settings(tmp_path):
    rio = pytest.importorskip("rasterio")
    path = tmp_path / "scene.tif"
    with rio.open(path, "w", driver="GTiff", height=96, width=112, count=3, dtype="float32") as dst:
        dst.write(scene())
    program = Program(batch_size=3, fixed_size=(64, 64))
    with pytest.warns(UserWarning, match="using patch_size=64"):
        written = predict_files(
            program,
            [path],
            output_dir=tmp_path / "out",
            load_func=lambda p: rio.open(p).read().astype(np.float32),
            patch_size=1000,
            overlap=300,
            **KW,
        )
    assert len(written) == 1
    assert {(s[0], *s[-2:]) for s in program.shapes} == {(3, 64, 64)}


def test_files_batch_mode_allows_a_non_square_size(tmp_path):
    program = Program(batch_size=2, fixed_size=(64, 96))
    chips = [scene(64, 96)]
    np.save(tmp_path / "a.npy", chips[0])
    predict_files(
        program,
        [tmp_path / "a.npy"],
        output_dir=tmp_path / "out",
        load_func=np.load,
        mode="batch",
        save_func=lambda prediction, meta, item, output_dir, **_: item,
        **KW,
    )
    assert program.shapes == [(2, 3, 64, 96)]


def test_the_programs_device_is_the_default():
    """No device=: a GPU machine would otherwise send patches to the GPU, and this conv is on
    the CPU."""
    program = Program(device="cpu")
    predict_array(program, scene(), patch_size=64, overlap=16, progress=False)
    assert program.shapes


def test_an_explicit_device_wins():
    assert program_device([Program(device="cpu")], "cuda") == "cuda"
    assert program_device([Program(device="cpu")], None) == torch.device("cpu")
    assert program_device([Program()], None) is None


def test_an_ensemble_must_agree():
    with pytest.raises(ValueError, match="different batch_size"):
        predict_array([Program(batch_size=2), Program(batch_size=3)], scene(), **TILED)
    with pytest.raises(ValueError, match="different fixed_size"):
        predict_array(
            [Program(fixed_size=(64, 64)), Program(fixed_size=(32, 32))], scene(), **TILED
        )


def test_an_ensemble_of_fixed_and_dynamic_programs_averages_them():
    out = predict_array([Program(batch_size=2), Program(seed=1)], scene(), **TILED)
    alone = [predict_array(Program(seed=seed), scene(), **TILED) for seed in (0, 1)]
    assert not np.allclose(alone[0], alone[1], atol=1e-3)
    np.testing.assert_allclose(out, (alone[0] + alone[1]) / 2, atol=1e-5)


def test_an_ensemble_compares_values_not_their_spelling():
    models = [Program(), Program()]
    models[0].device, models[1].device = "cpu", torch.device("cpu")
    models[0].fixed_size, models[1].fixed_size = [64, 64], (64, 64)
    assert predict_array(models, scene(), **TILED).shape == (2, 96, 112)
    assert all(shape[-2:] == (64, 64) for shape in models[0].shapes)


@pytest.mark.parametrize("spec", [{"max_size": 32}, {"pad_multiple": 48}, {"max_size": (32, 32)}])
def test_a_spec_cant_change_a_fixed_size(spec):
    """What the program takes wins over a spec that disagrees, so the patch stays the right size."""
    program = SpecProgram(spec, fixed_size=(64, 64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = predict_array(program, scene(), **TILED)
    assert out.shape == (2, 96, 112)
    assert {shape[-2:] for shape in program.shapes} == {(64, 64)}


def test_a_spec_given_as_an_argument_cant_change_a_fixed_size():
    program = Program(fixed_size=(64, 64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predict_array(program, scene(), spec={"max_size": 32}, **TILED)
    assert {shape[-2:] for shape in program.shapes} == {(64, 64)}


def test_an_impossible_overlap_is_refused_at_a_fixed_size():
    """The caller's own overlap: refused, as for any other model."""
    with pytest.raises(ValueError, match="overlap must be"):
        predict_array(Program(fixed_size=(64, 64)), scene(), patch_size=64, overlap=200, **KW)


def test_an_overlap_meant_for_another_patch_is_capped():
    with pytest.warns(UserWarning, match="overlap=32"):
        assert program_patch_size([Program(fixed_size=(64, 64))], 1000, 300, False) == (64, 32)
    assert program_patch_size([Program(fixed_size=(64, 64))], 64, 16, False) == (64, 16)


def test_modules_are_not_programs():
    """An nn.Module may use these names for something else, and they aren't read."""

    class Module(torch.nn.Module):
        batch_size, fixed_size, device = 3, (7, 9), "not a device"

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 3, padding=1)

        def forward(self, x):
            assert x.shape[0] == 2
            return self.conv(x)

    predict_array(Module(), scene(), batch_size=2, **TILED)


def test_capping_at_max_size_leaves_a_useful_stride():
    """The default 1000/300 on a 256 model was an overlap of 255: a stride of one pixel."""
    with pytest.warns(UserWarning, match="max_size"):
        assert fit_patch({"max_size": 256}, 1000, 300) == (256, 128)
    assert fit_patch({"max_size": 256}, 200, 100) == (200, 100)
