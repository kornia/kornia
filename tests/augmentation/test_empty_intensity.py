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

from copy import deepcopy

import pytest
import torch

import kornia.augmentation as K

from testing.base import BaseTester

_AUGMENTATIONS = [K.RandomEqualize, K.RandomEqualize3D, K.RandomPosterize, K.RandomSharpness]


def _shape(augmentation, batch=0):
    return (batch, 2, 3, 5, 7) if augmentation is K.RandomEqualize3D else (batch, 3, 6, 8)


def _assert_params_unchanged(actual, snapshot):
    assert actual.keys() == snapshot.keys()
    for name in actual:
        assert torch.equal(actual[name], snapshot[name]), name


class TestEmptyIntensityAugmentation(BaseTester):
    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    @pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_direct_callers_4429(self, augmentation, probability, same_on_batch, device, dtype):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype, requires_grad=True)
        aug = augmentation(p=probability, same_on_batch=same_on_batch).to(device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)
        # Exercise the function itself even when a container or the gate could bypass it.
        transformed = aug.apply_transform(image, params, aug.flags)
        assert transformed.shape == image.shape
        assert transformed.device == device
        assert transformed.dtype == dtype
        transformed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape
        image.grad = None
        output = aug(image, params=params)
        assert output.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        assert params["forward_input_shape"].tolist() == list(image.shape)
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    @pytest.mark.parametrize("state", ["copied", "after_forward", "fresh"])
    def test_saved_parameters_replay_4429(self, augmentation, state, device, dtype):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype, requires_grad=True)
        aug = augmentation(p=1).to(device=device, dtype=dtype)
        output = aug(image)
        params = deepcopy(aug._params)
        snapshot = deepcopy(params)
        if state == "after_forward":
            later_shape = (*_shape(augmentation, 2)[:-2], 9, 11)
            aug(torch.rand(later_shape).to(device=device, dtype=dtype))
        elif state == "fresh":
            aug = augmentation(p=1).to(device=device, dtype=dtype)
        cpu_rng = torch.random.get_rng_state().clone()
        replayed = aug(image, params=params)
        assert replayed.shape == image.shape
        assert replayed.dtype == dtype
        assert replayed.device == device
        self.assert_close(replayed, output, rtol=0, atol=0)
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        _assert_params_unchanged(params, snapshot)
        replayed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("container", ["image", "augmentation", "patch_same", "patch_valid"])
    def test_2d_container_replay_4429(self, container, device, dtype):
        children = [K.RandomEqualize(p=1), K.RandomPosterize(p=1), K.RandomSharpness(p=1)]
        if container == "image":
            sequence = K.ImageSequential(*children)
        elif container == "augmentation":
            sequence = K.AugmentationSequential(*children, data_keys=["input", "mask"])
        else:
            sequence = K.PatchSequential(
                *children, grid_size=(1, 3), padding=container.split("_")[1], patchwise_apply=True, random_apply=False
            )
        sequence = sequence.to(device=device, dtype=dtype)
        image = torch.empty(0, 3, 9, 13, device=device, dtype=dtype, requires_grad=True)
        if container == "augmentation":
            mask = torch.empty(0, 1, 9, 13, device=device, dtype=dtype)
            output, output_mask = sequence(image, mask)
            assert output_mask.shape == mask.shape
            replayed, replayed_mask = sequence(image, mask, params=deepcopy(sequence._params))
            assert replayed_mask.shape == mask.shape
        else:
            output = sequence(image)
            replayed = sequence(image, params=deepcopy(sequence._params))
        expected_shape = (0, 3, 9, 12 if container == "patch_valid" else 13)
        assert output.shape == replayed.shape == expected_shape
        assert output.device == device
        assert output.dtype == dtype
        output.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("layout", ["BTCHW", "BCTHW"])
    @pytest.mark.parametrize("same_on_frame", [False, True])
    def test_video_replay_4429(self, layout, same_on_frame, device, dtype):
        shape = (0, 2, 3, 6, 8) if layout == "BTCHW" else (0, 3, 2, 6, 8)
        video = torch.empty(shape, device=device, dtype=dtype, requires_grad=True)
        sequence = K.VideoSequential(
            K.RandomEqualize(p=1),
            K.RandomPosterize(p=1),
            K.RandomSharpness(p=1),
            data_format=layout,
            same_on_frame=same_on_frame,
        ).to(device=device, dtype=dtype)
        output = sequence(video)
        params = deepcopy(sequence._params)
        replayed = sequence(video, params=params)
        assert output.shape == replayed.shape == video.shape
        assert output.device == device
        assert output.dtype == dtype
        for item in params:
            assert item.data["forward_input_shape"][-2:].tolist() == [6, 8]
        replayed.sum().backward()
        assert video.grad is not None
        assert video.grad.shape == video.shape

    def test_3d_container_replay_4429(self, device, dtype):
        image = torch.empty(0, 2, 3, 5, 7, device=device, dtype=dtype, requires_grad=True)
        sequence = K.AugmentationSequential(K.RandomEqualize3D(p=1), data_keys=["input"]).to(device=device, dtype=dtype)
        output = sequence(image)
        replayed = sequence(image, params=deepcopy(sequence._params))
        assert output.shape == replayed.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        replayed.sum().backward()
        assert image.grad is not None
        assert image.grad.shape == image.shape

    @pytest.mark.parametrize("augmentation", _AUGMENTATIONS)
    def test_dynamo_direct_transform_4429(self, augmentation, device, dtype, torch_optimizer):
        image = torch.empty(_shape(augmentation), device=device, dtype=dtype)
        aug = augmentation(p=1).to(device=device, dtype=dtype)
        params = aug.forward_parameters(image.shape)

        def run(input):
            return aug.apply_transform(input, params, aug.flags)

        expected = run(image)
        compiled = torch_optimizer(run)
        output = compiled(image)
        assert output.shape == image.shape
        assert output.device == device
        assert output.dtype == dtype
        self.assert_close(output, expected)
