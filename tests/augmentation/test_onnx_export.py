# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ONNX export regression tests for kornia.augmentation modules.

These tests pin the set of augmentations that successfully export through
``torch.onnx.export`` (legacy TorchScript-tracer path) at opset 20, so future
changes to the augmentation base classes do not silently regress export
support.

Opset 20 was chosen as the floor because:
- ``aten::affine_grid_generator`` (used by every geometric augmentation that
  goes through ``warp_affine``) only lowers from opset 20 upward.
- ``aten::grid_sampler`` is supported from opset 16, well below 20.
- Earlier opsets work for non-geometric augmentations, but standardising on a
  single opset keeps the test matrix small.

What this file does NOT test:
- Numerical equivalence between the exported ONNX graph and the eager forward.
  That requires onnxruntime, which is not a hard dependency of kornia. Add a
  separate test file (e.g. ``test_onnx_export_numerical.py``) gated on the
  optional ``onnxruntime`` import when that work lands.
- The ``dynamo=True`` export path. That is tracked separately because it has
  different op coverage (notably, ``aten::linalg_solve`` lowers cleanly under
  dynamo but not under the legacy tracer).
- Augmentations that currently cannot export under the legacy tracer due to
  missing ONNX op support, see ``XFAIL_OPS`` below.

Adding an augmentation to ``EXPORTABLE_OPS`` is the way to advertise that it
cleanly exports today; adding to ``XFAIL_OPS`` marks a known gap with the
underlying op so we don't lose the signal that it's still pending.
"""

from __future__ import annotations

import io
import warnings
from typing import Any, Callable, Tuple

import pytest
import torch

import kornia.augmentation as K


def _hflip() -> torch.nn.Module:
    return K.RandomHorizontalFlip(p=1.0)


def _vflip() -> torch.nn.Module:
    return K.RandomVerticalFlip(p=1.0)


def _color_jiggle() -> torch.nn.Module:
    return K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)


def _brightness() -> torch.nn.Module:
    return K.RandomBrightness(brightness=(0.5, 1.5), p=1.0)


def _grayscale() -> torch.nn.Module:
    return K.RandomGrayscale(p=1.0)


def _invert() -> torch.nn.Module:
    return K.RandomInvert(p=1.0)


def _gaussian_blur() -> torch.nn.Module:
    return K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0)


def _erasing() -> torch.nn.Module:
    return K.RandomErasing(p=1.0)


def _posterize() -> torch.nn.Module:
    return K.RandomPosterize(3, p=1.0)


def _solarize() -> torch.nn.Module:
    return K.RandomSolarize(0.1, p=1.0)


def _normalize() -> torch.nn.Module:
    return K.Normalize(mean=torch.zeros(3), std=torch.ones(3))


def _affine() -> torch.nn.Module:
    return K.RandomAffine(degrees=10.0, p=1.0)


def _rotation() -> torch.nn.Module:
    return K.RandomRotation(degrees=10.0, p=1.0)


def _perspective() -> torch.nn.Module:
    return K.RandomPerspective(p=1.0)


def _augmentation_sequential() -> torch.nn.Module:
    """A torchgeo-typical pipeline: per-element random flips followed by normalize."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.Normalize(mean=torch.zeros(3), std=torch.ones(3)),
        data_keys=["input"],
    )


# Augmentations that export today through the legacy TorchScript tracer at opset 20.
EXPORTABLE_OPS: list[Tuple[str, Callable[[], torch.nn.Module]]] = [
    ("RandomHorizontalFlip", _hflip),
    ("RandomVerticalFlip", _vflip),
    ("ColorJiggle", _color_jiggle),
    ("RandomBrightness", _brightness),
    ("RandomGrayscale", _grayscale),
    ("RandomInvert", _invert),
    ("RandomGaussianBlur", _gaussian_blur),
    ("RandomErasing", _erasing),
    ("RandomPosterize", _posterize),
    ("RandomSolarize", _solarize),
    ("Normalize", _normalize),
    # Geometric augmentations enabled by:
    #   - bumping opset to 20 (for ``affine_grid_generator``)
    #   - the closed-form 3x3 inverse in ``kornia.core.utils._inverse_3x3_closed_form``
    #     that replaces ``aten::linalg_inv`` during ONNX export.
    ("RandomAffine", _affine),
    ("RandomRotation", _rotation),
    # ``RandomPerspective`` exports via the closed-form Heckbert decomposition in
    # ``kornia.geometry.transform.imgwarp._get_perspective_transform_closed_form``,
    # which replaces the 8x8 ``torch.linalg.solve`` with two unit-square-to-quad
    # constructions and one closed-form 3x3 inverse.
    ("RandomPerspective", _perspective),
    # Container: pinned because torchgeo and similar callers wrap their pipelines
    # in ``AugmentationSequential``. The container's ``forward`` strips trailing
    # ``None`` positional args injected by the legacy ONNX tracer.
    ("AugmentationSequential", _augmentation_sequential),
]

# Currently no augmentations are blocked at export time. Augmentations that export
# but produce numerically different results from eager are tracked separately in
# ``ONNX_NUMERICAL_KNOWN_DRIFT`` below.
XFAIL_OPS: list[Tuple[str, Callable[[], torch.nn.Module], str]] = []


def _try_export(module: torch.nn.Module, x: torch.Tensor) -> int:
    """Run ``torch.onnx.export`` against an in-memory buffer and return the byte count."""
    module.eval()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            (x,),
            buf,
            opset_version=20,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    return len(buf.getvalue())


@pytest.mark.parametrize("name,factory", EXPORTABLE_OPS, ids=[n for n, _ in EXPORTABLE_OPS])
def test_onnx_export_exportable(name: str, factory: Callable[[], torch.nn.Module]) -> None:
    """Augmentation exports cleanly via ``torch.onnx.export`` (legacy tracer, opset 20)."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    size = _try_export(factory(), x)
    assert size > 0, f"{name}: ONNX graph was empty"


@pytest.mark.parametrize("name,factory,reason", XFAIL_OPS, ids=[n for n, _, _ in XFAIL_OPS])
def test_onnx_export_known_blocked(
    name: str, factory: Callable[[], torch.nn.Module], reason: str
) -> None:
    """Augmentation cannot export today; pinned so we notice if it starts working."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    with pytest.raises(Exception):  # noqa: B017, onnx errors are not in a stable hierarchy
        _try_export(factory(), x)


# ----------------------------------------------------------------------------
# Numerical-equivalence checks (require onnxruntime).
#
# Each entry pins a concrete deterministic configuration of the augmentation is
# fixed angles, fixed brightness, etc. So comparing eager and ONNX is a
# clean apples-to-apples test of "did the export faithfully capture the
# computation". Augmentations with random parameter ranges are deliberately
# excluded from the numerical check because ``torch.onnx.export`` consumes RNG
# during tracing in ways that don't line up with a separate eager call, even
# under the same ``torch.manual_seed``, that mismatch is RNG bookkeeping and
# would mask real bugs. The export-success tests above cover the random case.
# ----------------------------------------------------------------------------

ort = pytest.importorskip("onnxruntime", reason="onnxruntime is required for numerical-equivalence checks")
np = pytest.importorskip("numpy")


# Deterministic factories for numerical tests. Tuple ranges of the form ``(v, v)``
# evaluate to a single fixed value under uniform sampling. Geometric augmentations
# additionally rely on the in-place-mutation fix in
# ``kornia.geometry.conversions.normal_transform_pixel`` and the ``deg2rad``
# inlining in the affine/shear ``compute_transformation`` paths.
def _hflip_det() -> torch.nn.Module:
    return K.RandomHorizontalFlip(p=1.0)


def _vflip_det() -> torch.nn.Module:
    return K.RandomVerticalFlip(p=1.0)


def _grayscale_det() -> torch.nn.Module:
    return K.RandomGrayscale(p=1.0)


def _invert_det() -> torch.nn.Module:
    return K.RandomInvert(p=1.0)


def _normalize_det() -> torch.nn.Module:
    return K.Normalize(mean=torch.zeros(3), std=torch.ones(3))


def _brightness_det() -> torch.nn.Module:
    return K.RandomBrightness(brightness=(1.2, 1.2), p=1.0)


def _color_jiggle_det() -> torch.nn.Module:
    return K.ColorJiggle(brightness=(1.2, 1.2), p=1.0)


def _gaussian_blur_det() -> torch.nn.Module:
    return K.RandomGaussianBlur((3, 3), (1.5, 1.5), p=1.0)


def _solarize_det() -> torch.nn.Module:
    return K.RandomSolarize(thresholds=(0.5, 0.5), additions=(0.0, 0.0), p=1.0)


def _rotation_det() -> torch.nn.Module:
    return K.RandomRotation(degrees=(15.0, 15.0), p=1.0)


def _affine_det() -> torch.nn.Module:
    return K.RandomAffine(
        degrees=(15.0, 15.0), translate=(0.0, 0.0), scale=(1.0, 1.0), shear=(0.0, 0.0), p=1.0
    )


def _affine_with_shear_det() -> torch.nn.Module:
    return K.RandomAffine(degrees=(0.0, 0.0), shear=(5.0, 5.0), p=1.0)


# Augmentations whose exported graph produces numerically equivalent results to
# eager when configured with deterministic parameters. ``max diff < 1e-3``
# threshold is conservative, most are float32 precision (< 1e-5).
ONNX_NUMERICAL_EQUIVALENT: list[Tuple[str, Callable[[], torch.nn.Module]]] = [
    # Parameter-free / no random sampling
    ("RandomHorizontalFlip", _hflip_det),
    ("RandomVerticalFlip", _vflip_det),
    ("RandomGrayscale", _grayscale_det),
    ("RandomInvert", _invert_det),
    ("Normalize", _normalize_det),
    # Random params pinned to deterministic ranges
    ("RandomBrightness", _brightness_det),
    ("ColorJiggle", _color_jiggle_det),
    ("RandomGaussianBlur", _gaussian_blur_det),
    ("RandomSolarize", _solarize_det),
    # Geometric augmentations, exercise the warp_affine / grid_sample path.
    # Previously diverged due to in-place mutation in normal_transform_pixel
    # losing the scale factors under tracing.
    ("RandomRotation", _rotation_det),
    ("RandomAffine", _affine_det),
    ("RandomAffine_with_shear", _affine_with_shear_det),
]


def _run_onnx(module: torch.nn.Module, x: torch.Tensor) -> Any:
    module.eval()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module, (x,), buf, opset_version=20, dynamo=False,
            input_names=["input"], output_names=["output"],
        )
    sess = ort.InferenceSession(buf.getvalue())
    return sess.run(["output"], {"input": x.numpy()})[0]


@pytest.mark.parametrize(
    "name,factory", ONNX_NUMERICAL_EQUIVALENT, ids=[n for n, _ in ONNX_NUMERICAL_EQUIVALENT]
)
def test_onnx_export_numerically_matches_eager(
    name: str, factory: Callable[[], torch.nn.Module]
) -> None:
    """Exported graph produces the same numbers as eager for the deterministic configuration."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    aug = factory()
    aug.eval()
    eager = aug(x).numpy()
    onnx_out = _run_onnx(aug, x)
    assert eager.shape == onnx_out.shape, f"{name}: shape mismatch {eager.shape} vs {onnx_out.shape}"
    max_diff = float(np.abs(eager - onnx_out).max())
    assert max_diff < 1e-3, f"{name}: max abs diff {max_diff:.4f} exceeds 1e-3"
