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

import pytest
import torch

import kornia
import kornia.augmentation as K

from testing.base import BaseTester


class TestDispatcher(BaseTester):
    @pytest.mark.parametrize("num_augmentations,num_inputs", [(2, 1), (2, 3), (2, 0), (0, 1)])
    def test_many_to_many_input_count(self, num_augmentations, num_inputs, device, dtype):
        image = torch.arange(48, device=device, dtype=dtype).reshape(2, 1, 4, 6)
        mask = image.remainder(2)
        augmentations = [
            K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"])
            for _ in range(num_augmentations)
        ]
        dispatcher = K.ManyToManyAugmentationDispather(*augmentations)

        message = f"Expected {num_augmentations} input bundles, one per augmentation, but got {num_inputs}\\."
        with pytest.raises(ValueError, match=message):
            dispatcher(*[(image, mask)] * num_inputs)

        # Validate before applying any augmentation or updating its replay parameters.
        assert all(augmentation._params is None for augmentation in augmentations)

    @pytest.mark.parametrize("bundle_type", [tuple, list])
    def test_many_to_many_corresponding_inputs(self, bundle_type, device, dtype):
        image_1 = torch.arange(48, device=device, dtype=dtype).reshape(2, 1, 4, 6)
        image_2 = image_1 + 48
        mask = image_2.remainder(3)
        dispatcher = K.ManyToManyAugmentationDispather(
            K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input"]),
            K.AugmentationSequential(K.RandomVerticalFlip(p=1.0), data_keys=["input", "mask"]),
        )

        output = dispatcher(bundle_type([image_1]), bundle_type([image_2, mask]))

        assert len(output) == 2
        self.assert_close(output[0], image_1.flip(-1), rtol=0, atol=0)
        self.assert_close(output[1][0], image_2.flip(-2), rtol=0, atol=0)
        self.assert_close(output[1][1], mask.flip(-2), rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_many_to_many_empty(self):
        # Preserve current behavior; rejecting an empty dispatcher remains under discussion in #4422.
        assert K.ManyToManyAugmentationDispather()() == []

    @pytest.mark.parametrize("strict", [True, False])
    def test_many_to_one(self, strict, device, dtype):
        # Each container gets the same bundle and returns its own result, so the two flips must both show.
        input = torch.arange(60, device=device, dtype=dtype).reshape(2, 1, 5, 6)
        mask = input.remainder(3)
        aug_list = K.ManyToOneAugmentationDispather(
            K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "mask"]),
            K.AugmentationSequential(K.RandomVerticalFlip(p=1.0), data_keys=["input", "mask"]),
            strict=strict,
        )
        output = aug_list(input, mask)

        assert len(output) == 2
        self.assert_close(output[0][0], input.flip(-1), rtol=0, atol=0)
        self.assert_close(output[0][1], mask.flip(-1), rtol=0, atol=0)
        self.assert_close(output[1][0], input.flip(-2), rtol=0, atol=0)
        self.assert_close(output[1][1], mask.flip(-2), rtol=0, atol=0)

    @pytest.mark.parametrize("strict", [True, False])
    def test_many_to_one_strict_mode(self, strict):
        def _init_many_to_one(strict):
            K.ManyToOneAugmentationDispather(
                K.AugmentationSequential(
                    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                    kornia.augmentation.RandomAffine(360, p=1.0),
                    data_keys=["input"],
                ),
                K.AugmentationSequential(
                    kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                    kornia.augmentation.RandomAffine(360, p=1.0),
                    data_keys=["input", "mask"],
                ),
                strict=strict,
            )

        if strict:
            with pytest.raises(RuntimeError):
                _init_many_to_one(strict)  # fails
        else:
            _init_many_to_one(strict)  # passes
