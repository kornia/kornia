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
from torch.testing import assert_close

from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation


class PresetTests:
    pass


@pytest.mark.usefixtures("device", "dtype")
class TestAdaptiveDiscriminatorAugmentation(PresetTests):
    def test_initial_hyper_params(self):
        ada_preset = AdaptiveDiscriminatorAugmentation()
        assert ada_preset.adjustment_speed > 0
        assert 0 <= ada_preset.target_real_acc <= 1
        assert 0 <= ada_preset.ema_lambda <= 1
        assert ada_preset.update_every >= 1
        assert 0 <= ada_preset.max_p <= 1
        assert 0 <= ada_preset.p <= ada_preset.max_p  # initial p
        assert ada_preset.real_acc_ema == 0.5
        assert ada_preset.num_calls == 0

        transforms = list(ada_preset.children())
        expected_transforms = [
            "RandomHorizontalFlip",
            "RandomRotation90",
            "RandomErasing",
            "RandomAffine",
            "ColorJitter",
            "RandomGaussianNoise",
        ]

        assert len(transforms) == len(expected_transforms)
        for t, et in zip(transforms, expected_transforms):
            assert et == str(t.__class__.__name__)

    def test_transforms_behaviour(self, device, dtype):
        ada_preset = AdaptiveDiscriminatorAugmentation().to(device)
        inputs = torch.randn(2, 3, 32, 32).to(device)
        outputs = ada_preset(inputs)
        assert outputs.dtype == inputs.dtype
        assert outputs.shape == inputs.shape

        ada_preset.p = 0
        ada_outputs = ada_preset(inputs)
        assert_close(inputs, ada_outputs)
