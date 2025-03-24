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


class TestDispatcher:
    def test_many_to_many(self, device, dtype):
        input_1 = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        input_2 = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        mask_1 = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        mask_2 = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        aug_list = K.ManyToManyAugmentationDispather(
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
        )
        output = aug_list((input_1, mask_1), (input_2, mask_2))

        assert output[0][0].shape == input_1.shape
        assert output[1][0].shape == input_2.shape
        assert output[0][1].shape == mask_1.shape
        assert output[1][1].shape == mask_2.shape

    @pytest.mark.parametrize("strict", [True, False])
    def test_many_to_one(self, strict, device, dtype):
        input = torch.randn(2, 3, 5, 6, device=device, dtype=dtype)
        mask = torch.ones(2, 1, 5, 6, device=device, dtype=dtype)
        aug_list = K.ManyToOneAugmentationDispather(
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            K.AugmentationSequential(
                kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
                kornia.augmentation.RandomAffine(360, p=1.0),
                data_keys=["input", "mask"],
            ),
            strict=strict,
        )
        output = aug_list(input, mask)

        assert output[0][0].shape == input.shape
        assert output[1][0].shape == input.shape
        assert output[0][1].shape == mask.shape
        assert output[1][1].shape == mask.shape

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
