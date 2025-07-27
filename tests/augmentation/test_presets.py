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

from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation

from testing.base import BaseTester


class PresetTests(BaseTester):
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
        assert ada_preset._num_calls == 0
        self.assert_close(ada_preset.real_acc_ema, 0.5)

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
        self.assert_close(inputs, ada_outputs)

    def test_adaptive_probability(self, device, dtype):
        inputs = torch.randn(2, 3, 32, 32)
        n_runs = 3
        initial_p = 0.5
        update_every = 3
        ada = AdaptiveDiscriminatorAugmentation(
            initial_p=initial_p,
            adjustment_speed=0.01,
            max_p=0.8,
            update_every=update_every,
            target_real_acc=0.9,
            ema_lambda=0,
        )

        # p increasing, without reaching max_p
        for i in range(ada.update_every * n_runs):
            self.assert_close(ada.p, initial_p + (i // update_every) * ada.adjustment_speed)
            ada(inputs, real_acc=ada.target_real_acc + 0.1)
        self.assert_close(ada.p, initial_p + n_runs * ada.adjustment_speed)

        # decreasing without reaching 0
        initial_p = ada.p
        for i in range(ada.update_every * n_runs):
            self.assert_close(ada.p, initial_p - (i // update_every) * ada.adjustment_speed)
            ada(inputs, real_acc=ada.target_real_acc - 0.1)
        self.assert_close(ada.p, initial_p - n_runs * ada.adjustment_speed)

        # p clamped at 0
        ada.p = ada.adjustment_speed / 2
        for _ in range(ada.update_every):
            ada(inputs, real_acc=ada.target_real_acc - 0.1)
        self.assert_close(ada.p, 0)

        # p clamped at max_p
        ada.p = ada.max_p - ada.adjustment_speed / 2
        for _ in range(ada.update_every):
            ada(inputs, real_acc=ada.target_real_acc + 0.1)
        self.assert_close(ada.p, ada.max_p)
