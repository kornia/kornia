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

"""Inventory empty image batches without inferring coverage from a selected list of regressions (#4429)."""

import copy
import functools
import inspect
from unittest.mock import Mock

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation.base import _BasicAugmentationBase

from testing.augmentation.sync_registry import AUG_CASES, EXCLUDED, AugCase
from testing.base import BaseTester

_CASES = [case for case in AUG_CASES if case.kind in ("2d", "3d", "mix")]
# The empty-image contract bypasses diffusion inference, so it can be checked without model downloads.
_CASES.append(AugCase("RandomDissolving", functools.partial(K.RandomDissolving), "2d", ("forward",)))
# These non-default configurations were reported separately in #4429; the default factories do not exercise them.
_CASES.extend(
    [
        AugCase(
            "RandomMotionBlur[ranged]", functools.partial(K.RandomMotionBlur, (3, 7), 35.0, 0.5), "2d", ("forward",)
        ),
        AugCase(
            "RandomMotionBlur3D[ranged]", functools.partial(K.RandomMotionBlur3D, (3, 5), 35.0, 0.5), "3d", ("forward",)
        ),
        AugCase(
            "RandomMotionBlur3D[bilinear]",
            functools.partial(K.RandomMotionBlur3D, 3, 35.0, 0.5, resample="bilinear"),
            "3d",
            ("forward",),
        ),
    ]
)

# Expected sizes of the registry's explicit constructor configurations, independent of sampled parameters.
_OUTPUT_SPATIAL = {
    "CenterCrop": (16, 16),
    "LongestMaxSize": (24, 24),
    "PadTo": (40, 40),
    "RandomCrop": (16, 16),
    "RandomResizedCrop": (16, 16),
    "Resize": (24, 24),
    "SmallestMaxSize": (24, 24),
    "CenterCrop3D": (4, 8, 8),
    "RandomCrop3D": (4, 8, 8),
}


class TestEmptyBatchInventory(BaseTester):
    def test_public_concrete_classes_are_inventoried(self):
        exported = {
            name
            for name in K.__all__
            if inspect.isclass(cls := getattr(K, name))
            and issubclass(cls, _BasicAugmentationBase)
            and EXCLUDED.get(name) != "abstract base"
        }
        assert {case.name.partition("[")[0] for case in _CASES} == exported

    @pytest.mark.parametrize(
        "case,probability",
        [
            pytest.param(case, probability, id=f"{case.name}-{probability}")
            for case in _CASES
            for probability in ((None,) if case.name == "PadTo" else (0.0, 0.5, 1.0))
        ],
    )
    def test_empty_forward_and_saved_parameters(self, case, probability, device, dtype, monkeypatch):
        if case.name == "RandomDissolving":
            model = torch.nn.Identity()
            monkeypatch.setattr(
                model, "forward", Mock(side_effect=AssertionError("Empty images must bypass inference"))
            )
            monkeypatch.setattr(
                "kornia.augmentation._2d.intensity.dissolving.StableDiffusionDissolving", Mock(return_value=model)
            )
        augmentation = case.build() if probability is None else case.build(p=probability)
        spatial = (8, 16, 16) if case.kind == "3d" else (32, 32)
        image = torch.empty(0, 3, *spatial, device=device, dtype=dtype)
        inputs = [image]
        expected_shapes = [(0, 3, *_OUTPUT_SPATIAL.get(case.name.partition("[")[0], spatial))]
        parameter_shape = image.shape
        if isinstance(augmentation, (K.RandomTransplantation, K.RandomTransplantation3D)):
            inputs.append(torch.empty(0, *spatial, device=device, dtype=torch.long))
            expected_shapes.append(tuple(inputs[-1].shape))
            parameter_shape = inputs[-1].shape  # Transplantation samples from its driving mask.
        elif "class" in case.inputs:
            inputs.append(torch.empty(0, device=device, dtype=torch.long))
            expected_shapes.append((1, 0, 3) if isinstance(augmentation, K.RandomCutMixV2) else (0, 3))

        output = augmentation(*inputs)
        params = copy.deepcopy(augmentation._params)
        assert params["batch_prob"].numel() == 0
        assert params["forward_input_shape"].tolist() == list(parameter_shape)

        def no_parameter_draw(*args, **kwargs):
            pytest.fail("Saved empty parameters must not be sampled again")

        monkeypatch.setattr(augmentation, "forward_parameters", no_parameter_draw)
        replay = augmentation(*inputs, params=params)
        for actual in (output, replay):
            tensors = [actual] if isinstance(actual, torch.Tensor) else actual
            assert len(tensors) == len(expected_shapes)
            for tensor, expected_shape in zip(tensors, expected_shapes):
                assert tensor.shape == expected_shape
                assert tensor.device == device
                assert tensor.numel() == 0
            assert tensors[0].dtype == dtype
