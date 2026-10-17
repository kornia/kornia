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

"""Registry of augmentation calls whose CUDA host synchronizations are counted.

Every public augmentation, container, automatic policy and preset is listed here with explicit constructor
arguments, together with the public functions in which a host synchronization was found. ``make_call`` builds
the inputs and returns a closure that runs one case, so a caller can warm it up and then count what it
synchronizes with ``sync_sites``. The CUDA guard in ``tests/augmentation/test_host_sync.py`` and the
byte-identity snapshots both read this file, so it uses only kornia's public API and torch.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import os
import traceback
import warnings
from typing import Any, Callable

import torch

import kornia
import kornia.augmentation as K
from kornia.augmentation import auto as KAuto
from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation

__all__ = ["AUG_CASES", "EXCLUDED", "INHERENT_SITES", "PASS_THROUGH", "AugCase", "make_call", "sync_sites"]

# The message torch emits for every synchronizing call under ``torch.cuda.set_sync_debug_mode("warn")``.
_SYNC_MESSAGE = "called a synchronizing CUDA operation"

ALL_2D = ("forward", "seq_annotations", "p_half")
GEOMETRIC_2D = ("forward", "inverse", "seq_annotations", "seq_inverse", "p_half")
FORWARD_P = ("forward", "p_half")


@dataclasses.dataclass(frozen=True)
class AugCase:
    """One registered augmentation, container, policy or function.

    Attributes:
        name: Public name, used in the test ids.
        build: Zero-argument factory. For augmentations it is a ``functools.partial`` of the class, so the
            ``p_half`` case can rebuild it with ``p=0.5``. For ``"functional"`` entries it returns a function
            that takes the 2D input and returns the closure to run.
        kind: One of ``"2d"``, ``"3d"``, ``"mix"``, ``"container"``, ``"auto"`` and ``"functional"``.
        cases: The calls to make, a subset of ``"forward"``, ``"inverse"``, ``"seq_annotations"``,
            ``"seq_inverse"`` and ``"p_half"``.
    """

    name: str
    build: Callable[[], Any]
    kind: str
    cases: tuple[str, ...]


def _aug(cls: type, *args: Any, kind: str = "2d", cases: tuple[str, ...] = ALL_2D, **kwargs: Any) -> AugCase:
    return AugCase(cls.__name__, functools.partial(cls, *args, **kwargs), kind, cases)


def _functional(name: str, setup: Callable[[torch.Tensor], Callable[[], Any]]) -> AugCase:
    return AugCase(name, lambda: setup, "functional", ("forward",))


def _setup_rotate(x: torch.Tensor) -> Callable[[], Any]:
    angle = torch.linspace(-30.0, 30.0, x.shape[0], device=x.device, dtype=x.dtype)
    return lambda: kornia.geometry.transform.rotate(x, angle)


def _setup_elastic(x: torch.Tensor) -> Callable[[], Any]:
    noise = torch.rand(x.shape[0], 2, *x.shape[-2:], device=x.device, dtype=x.dtype) * 2 - 1
    return lambda: kornia.geometry.transform.elastic_transform2d(x, noise)


def _setup_jpeg(x: torch.Tensor) -> Callable[[], Any]:
    # A CPU quality, as RandomJPEG passes: the range check of a CUDA quality reads it back on the host.
    quality = torch.linspace(50.0, 90.0, x.shape[0], dtype=x.dtype)
    return lambda: kornia.enhance.jpeg_codec_differentiable(x, quality)


def _setup_crop_by_indices(x: torch.Tensor) -> Callable[[], Any]:
    # Augmentations pass boxes sampled on the CPU, so the functional entry does too.
    boxes = torch.tensor([[[2, 3], [17, 3], [17, 18], [2, 18]]], dtype=torch.long).repeat(x.shape[0], 1, 1)
    return lambda: kornia.geometry.transform.crop_by_indices(x, boxes, size=(16, 16))


FUNCTIONAL_CASES = [
    _functional(
        "normal_transform_pixel",
        lambda x: lambda: kornia.geometry.conversions.normal_transform_pixel(32, 32, device=x.device, dtype=x.dtype),
    ),
    _functional(
        "normal_transform_pixel3d",
        lambda x: lambda: kornia.geometry.conversions.normal_transform_pixel3d(
            8, 16, 16, device=x.device, dtype=x.dtype
        ),
    ),
    _functional(
        "get_gaussian_kernel1d",
        lambda x: functools.partial(
            kornia.filters.get_gaussian_kernel1d,
            5,
            torch.full((x.shape[0], 1), 1.5, device=x.device, dtype=x.dtype),
        ),
    ),
    _functional("rgb_to_grayscale", lambda x: functools.partial(kornia.color.rgb_to_grayscale, x)),
    _functional("rgb_to_hls", lambda x: functools.partial(kornia.color.rgb_to_hls, x)),
    _functional("hls_to_rgb", lambda x: functools.partial(kornia.color.hls_to_rgb, x)),
    _functional("rotate", _setup_rotate),
    _functional("elastic_transform2d", _setup_elastic),
    _functional("sharpness", lambda x: functools.partial(kornia.enhance.sharpness, x, 0.5)),
    _functional("jpeg_codec_differentiable", _setup_jpeg),
    _functional("gaussian_blur2d", lambda x: functools.partial(kornia.filters.gaussian_blur2d, x, (5, 5), (1.5, 1.5))),
    _functional("crop_by_indices", _setup_crop_by_indices),
]


def _intensity_cases() -> list[AugCase]:
    return [
        _aug(K.ColorJiggle, 0.1, 0.1, 0.1, 0.1, p=1.0),
        _aug(K.ColorJitter, 0.1, 0.1, 0.1, 0.1, p=1.0),
        _aug(K.Denormalize, mean=(0.5, 0.4, 0.3), std=(0.2, 0.25, 0.3)),
        _aug(K.Normalize, mean=(0.5, 0.4, 0.3), std=(0.2, 0.25, 0.3)),
        _aug(K.RandomAutoContrast, p=1.0),
        _aug(K.RandomBoxBlur, p=1.0),
        _aug(K.RandomBrightness, brightness=(0.8, 1.2), p=1.0),
        _aug(K.RandomChannelDropout, p=1.0),
        _aug(K.RandomChannelShuffle, p=1.0),
        _aug(K.RandomClahe, p=1.0),
        _aug(K.RandomContrast, contrast=(0.8, 1.2), p=1.0),
        _aug(K.RandomEqualize, p=1.0),
        _aug(K.RandomErasing, p=1.0),
        _aug(K.RandomGamma, gamma=(0.8, 1.2), gain=(0.9, 1.1), p=1.0),
        _aug(K.RandomGaussianBlur, (5, 5), (0.1, 2.0), p=1.0),
        _aug(K.RandomGaussianIllumination, p=1.0),
        _aug(K.RandomGaussianNoise, std=0.1, p=1.0),
        _aug(K.RandomGrayscale, p=1.0),
        _aug(K.RandomHue, hue=(-0.1, 0.1), p=1.0),
        _aug(K.RandomInvert, p=1.0),
        _aug(K.RandomJPEG, jpeg_quality=(50.0, 90.0), p=1.0),
        _aug(K.RandomLinearCornerIllumination, p=1.0),
        _aug(K.RandomLinearIllumination, p=1.0),
        _aug(K.RandomMedianBlur, p=1.0),
        _aug(K.RandomMotionBlur, 3, 35.0, 0.5, p=1.0),
        _aug(K.RandomPlanckianJitter, p=1.0),
        _aug(K.RandomPlasmaBrightness, p=1.0),
        _aug(K.RandomPlasmaContrast, p=1.0),
        _aug(K.RandomPlasmaShadow, p=1.0),
        _aug(K.RandomPosterize, p=1.0),
        _aug(K.RandomRGBShift, p=1.0),
        _aug(K.RandomRain, p=1.0),
        _aug(K.RandomSaltAndPepperNoise, p=1.0),
        _aug(K.RandomSaturation, saturation=(0.8, 1.2), p=1.0),
        _aug(K.RandomSharpness, p=1.0),
        _aug(K.RandomSnow, p=1.0),
        _aug(K.RandomSolarize, p=1.0),
    ]


def _geometric_cases() -> list[AugCase]:
    fisheye = {
        "center_x": torch.tensor([-0.3, 0.3]),
        "center_y": torch.tensor([-0.3, 0.3]),
        "gamma": torch.tensor([0.9, 1.0]),
    }
    return [
        _aug(K.CenterCrop, 16),
        _aug(K.LongestMaxSize, 24),
        _aug(K.PadTo, (40, 40), cases=("forward", "inverse", "seq_annotations", "seq_inverse")),
        _aug(K.RandomAffine, 15.0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10.0, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomCrop, (16, 16), p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomElasticTransform, p=1.0),
        _aug(K.RandomFisheye, **fisheye, p=1.0, cases=FORWARD_P),
        _aug(K.RandomHorizontalFlip, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomPerspective, 0.5, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomResizedCrop, (16, 16), p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomRotation, 30.0, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomRotation90, (0, 3), p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomShear, 10.0, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomThinPlateSpline, p=1.0, cases=FORWARD_P),
        _aug(K.RandomTranslate, (-0.1, 0.1), (-0.1, 0.1), p=1.0, cases=GEOMETRIC_2D),
        _aug(K.RandomVerticalFlip, p=1.0, cases=GEOMETRIC_2D),
        _aug(K.Resize, (24, 24), cases=GEOMETRIC_2D),
        _aug(K.SmallestMaxSize, 24, cases=GEOMETRIC_2D),
    ]


def _mix_cases() -> list[AugCase]:
    return [
        _aug(K.PatchMix, patch_size=8, p=1.0, kind="mix", cases=FORWARD_P),
        _aug(K.RandomCutMixV2, p=1.0, use_correct_lambda=True, kind="mix", cases=FORWARD_P),
        _aug(K.RandomJigsaw, (4, 4), p=1.0, kind="mix", cases=FORWARD_P),
        _aug(K.RandomMixUpV2, p=1.0, kind="mix", cases=FORWARD_P),
        _aug(K.RandomMosaic, (32, 32), p=1.0, kind="mix", cases=FORWARD_P),
        _aug(K.RandomTransplantation, p=1.0, kind="mix", cases=FORWARD_P),
    ]


def _cases_3d() -> list[AugCase]:
    return [
        _aug(K.CenterCrop3D, (4, 8, 8), kind="3d", cases=FORWARD_P),
        _aug(K.RandomAffine3D, 15.0, translate=(0.1, 0.1, 0.1), scale=(0.9, 1.1), p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomCrop3D, (4, 8, 8), p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomDepthicalFlip3D, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomEqualize3D, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomHorizontalFlip3D, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomMotionBlur3D, 3, 35.0, 0.5, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomPerspective3D, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomRotation3D, 15.0, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomTransplantation3D, p=1.0, kind="3d", cases=FORWARD_P),
        _aug(K.RandomVerticalFlip3D, p=1.0, kind="3d", cases=FORWARD_P),
    ]


def _pipeline() -> list[torch.nn.Module]:
    return [
        K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        K.RandomAffine(15.0, translate=(0.1, 0.1), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
    ]


def _container_cases() -> list[AugCase]:
    def augmentation_sequential() -> K.AugmentationSequential:
        return K.AugmentationSequential(*_pipeline())

    def image_sequential() -> K.ImageSequential:
        return K.ImageSequential(*_pipeline(), random_apply=2)

    def video_sequential() -> K.VideoSequential:
        return K.VideoSequential(*_pipeline(), data_format="BCTHW", same_on_frame=True)

    def patch_sequential() -> K.PatchSequential:
        return K.PatchSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomAffine(15.0, p=1.0),
            grid_size=(2, 2),
            patchwise_apply=False,
        )

    def ada() -> AdaptiveDiscriminatorAugmentation:
        return AdaptiveDiscriminatorAugmentation(initial_p=0.8)

    return [
        AugCase("AugmentationSequential", augmentation_sequential, "container", ("forward", "inverse")),
        AugCase("ImageSequential", image_sequential, "container", ("forward",)),
        AugCase("VideoSequential", video_sequential, "container", ("forward",)),
        AugCase("PatchSequential", patch_sequential, "container", ("forward",)),
        AugCase("AdaptiveDiscriminatorAugmentation", ada, "container", ("forward",)),
    ]


# Every operation the automatic policies draw from, each run on its own so that a fixed seed covers all of them.
_AUTO_OPERATIONS = [
    ("auto_contrast", 0, 1),
    ("equalize", 0, 1),
    ("invert", 0, 1),
    ("rotate", -30.0, 30.0),
    ("posterize", 4.0, 8.0),
    ("solarize", 0.0, 1.0),
    ("solarize_add", 0.0, 0.43),
    ("color", 0.1, 1.9),
    ("contrast", 0.1, 1.9),
    ("brightness", 0.1, 1.9),
    ("sharpness", 0.1, 1.9),
    ("shear_x", -0.3, 0.3),
    ("shear_y", -0.3, 0.3),
    ("translate_x", -0.1, 0.1),
    ("translate_y", -0.1, 0.1),
]


def _auto_cases() -> list[AugCase]:
    cases = [
        AugCase("AutoAugment", KAuto.AutoAugment, "auto", ("forward",)),
        AugCase("RandAugment", functools.partial(KAuto.RandAugment, n=2, m=10), "auto", ("forward",)),
        AugCase("TrivialAugment", KAuto.TrivialAugment, "auto", ("forward",)),
    ]
    for op in _AUTO_OPERATIONS:
        build = functools.partial(KAuto.RandAugment, n=1, m=10, policy=[[op]])
        cases.append(AugCase(f"RandAugment[{op[0]}]", build, "auto", ("forward",)))
    return cases


AUG_CASES: list[AugCase] = [
    *_intensity_cases(),
    *_geometric_cases(),
    *_mix_cases(),
    *_cases_3d(),
    *_container_cases(),
    *_auto_cases(),
    *FUNCTIONAL_CASES,
]

EXCLUDED: dict[str, str] = {
    "RandomDissolving": "downloads a Stable Diffusion model on first use",
    "AugmentationBase2D": "abstract base",
    "AugmentationBase3D": "abstract base",
    "GeometricAugmentationBase2D": "abstract base",
    "GeometricAugmentationBase3D": "abstract base",
    "IntensityAugmentationBase2D": "abstract base",
    "IntensityAugmentationBase3D": "abstract base",
    "MixAugmentationBaseV2": "abstract base",
    "RigidAffineAugmentationBase2D": "abstract base",
    "RigidAffineAugmentationBase3D": "abstract base",
}

# ``"relpath.py:function"`` keys of the synchronizations that stay, each with the reason it is inherent.
INHERENT_SITES: dict[str, str] = {
    "geometry/boxes.py:_validate_box_values": (
        "validation of user boxes: the non-finite, width and height errors read the box data on the host, in one "
        "transfer for all three checks"
    ),
    "augmentation/_2d/mix/transplantation.py:params_from_input": (
        "the transplanted labels are drawn from the values of the mask, which lives on the device (`unique` has a "
        "data-dependent size)"
    ),
    "augmentation/presets/ada.py:forward": (
        "ADA draws the augmented subset on the input's device and returns early when it is empty"
    ),
    "augmentation/presets/ada.py:_sample_inputs": (
        "ADA augments the drawn subset only, whose size depends on the device draw (boolean indexing)"
    ),
    "augmentation/presets/ada.py:_merge_inputs": "ADA writes the augmented subset back by its device mask",
    "geometry/transform/thin_plate_spline.py:get_tps_transform": (
        "`torch.linalg.solve` checks the user-visible TPS system for singularity and raises (decision D1)"
    ),
}

# Thin wrappers whose synchronizations belong to their caller. A site is the deepest kornia frame that is not one of
# them, so that INHERENT_SITES names the caller, which is what decides whether the synchronization is inherent.
PASS_THROUGH: frozenset[str] = frozenset({"core/utils.py:_torch_inverse_cast", "core/utils.py:_torch_solve_cast"})


def _inputs_2d(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    boxes = torch.tensor([[[2.0, 3.0, 20.0, 25.0]]], device=device, dtype=dtype).repeat(4, 1, 1)
    keypoints = torch.tensor([[[5.0, 6.0], [20.0, 21.0]]], device=device, dtype=dtype).repeat(4, 1, 1)
    return {
        "input": torch.rand(4, 3, 32, 32, device=device, dtype=dtype),
        "mask": (torch.rand(4, 1, 32, 32, device=device) > 0.5).to(dtype),
        "labels": torch.randint(0, 3, (4, 32, 32), device=device),
        "bbox_xyxy": boxes,
        "keypoints": keypoints,
    }


def _inputs_3d(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "input": torch.rand(2, 3, 8, 16, 16, device=device, dtype=dtype),
        "labels": torch.randint(0, 3, (2, 8, 16, 16), device=device),
    }


def _build(case: AugCase, which: str) -> Any:
    if which == "p_half":
        build = case.build
        if not isinstance(build, functools.partial):
            raise ValueError(f"{case.name} has no constructor arguments to rebuild with p=0.5")
        return functools.partial(build.func, *build.args, **{**build.keywords, "p": 0.5})()
    return case.build()


def make_call(case: AugCase, which: str, device: torch.device, dtype: torch.dtype) -> Callable[[], Any]:
    """Build the module and inputs of ``case`` on ``device`` and return a closure that runs ``which``.

    The closure carries the module it calls as its ``module`` attribute (``None`` for functions), so a caller
    can read ``_params`` and ``transform_matrix`` after running it, and its input tensors by data key as its
    ``inputs`` attribute, so a caller can make the image require gradients before running it.
    """
    if which not in case.cases:
        raise ValueError(f"{case.name} does not register the case {which!r}")
    module = _build(case, which)
    fn: Callable[[], Any]
    if case.kind == "functional":
        data = _inputs_2d(device, dtype)
        fn = module(data["input"])
        fn.module = None  # type: ignore[attr-defined]
        fn.inputs = data  # type: ignore[attr-defined]
        return fn

    if case.kind == "3d":
        data = _inputs_3d(device, dtype)
        x = data["input"]
        if isinstance(module, K.RandomTransplantation3D):
            labels = data["labels"]

            def fn() -> Any:
                return module(x, labels)
        else:

            def fn() -> Any:
                return module(x)
    elif case.kind == "container" and isinstance(module, K.VideoSequential):
        data = {"input": torch.rand(2, 3, 4, 32, 32, device=device, dtype=dtype)}
        x = data["input"]

        def fn() -> Any:
            return module(x)
    else:
        data = _inputs_2d(device, dtype)
        x, mask, boxes, keypoints = data["input"], data["mask"], data["bbox_xyxy"], data["keypoints"]
        if which in ("forward", "p_half"):
            if isinstance(module, K.RandomTransplantation):
                labels = data["labels"]

                def fn() -> Any:
                    return module(x, labels)
            else:

                def fn() -> Any:
                    return module(x)
        elif which == "inverse":

            def fn() -> Any:
                return module.inverse(module(x))
        else:
            seq = K.AugmentationSequential(module, data_keys=["input", "mask", "bbox_xyxy", "keypoints"])
            if which == "seq_annotations":

                def fn() -> Any:
                    return seq(x, mask, boxes, keypoints)
            else:

                def fn() -> Any:
                    return seq.inverse(*seq(x, mask, boxes, keypoints))

    fn.module = module  # type: ignore[attr-defined]
    fn.inputs = data  # type: ignore[attr-defined]
    return fn


def sync_sites(fn: Callable[[], Any]) -> collections.Counter[str]:
    """Run ``fn`` under the CUDA sync debug mode and count its synchronizations by site.

    A site is the deepest kornia frame on the stack that is not in ``PASS_THROUGH``, as ``"relpath.py:function"``
    relative to the kornia package; a synchronization with no kornia frame on the stack counts as
    ``"<outside kornia>"``. The debug mode is a prototype that misses some synchronizations, so a profiler count of
    ``cudaStreamSynchronize`` is the cross-check. The mode is reset to ``0`` even when ``fn`` raises.
    """
    root = os.path.dirname(kornia.__file__) + os.sep
    sites: collections.Counter[str] = collections.Counter()

    def hook(message: Any, category: Any, filename: str, lineno: int, file: Any = None, line: Any = None) -> None:
        if _SYNC_MESSAGE not in str(message):
            return
        keys = [f"{f.filename[len(root) :]}:{f.name}" for f in traceback.extract_stack() if f.filename.startswith(root)]
        callers = [k for k in keys if k not in PASS_THROUGH]
        sites[callers[-1] if callers else keys[-1] if keys else "<outside kornia>"] += 1

    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = hook
            fn()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    return sites
