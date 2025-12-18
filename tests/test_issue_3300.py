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

import torch

import kornia.augmentation as K


class TestIssue3300MixupOnlyCrash:
    """Test for issue #3300: Crash when only MixUp and CutMix are in AugmentationSequential."""

    def test_mixup_only_augmentation_sequential(self):
        """Test that AugmentationSequential works with only mix augmentations."""
        augseq = K.AugmentationSequential(
            K.RandomMixUpV2(p=1.0),
            random_apply=1,
        )
        x = torch.rand(4, 3, 32, 32, dtype=torch.float32)
        y = augseq(x)
        assert y.shape == x.shape

    def test_cutmix_only_augmentation_sequential(self):
        """Test that AugmentationSequential works with only CutMix."""
        augseq = K.AugmentationSequential(
            K.RandomCutMixV2(p=1.0),
            random_apply=1,
        )
        x = torch.rand(4, 3, 32, 32, dtype=torch.float32)
        y = augseq(x)
        assert y.shape == x.shape

    def test_multiple_mix_only_augmentation_sequential(self):
        """Test that AugmentationSequential works with only MixUp and CutMix."""
        augseq = K.AugmentationSequential(
            K.RandomMixUpV2(p=1.0),
            K.RandomCutMixV2(p=1.0),
            random_apply=1,
        )
        x = torch.rand(4, 3, 32, 32, dtype=torch.float32)
        y = augseq(x)
        assert y.shape == x.shape
