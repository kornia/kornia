import pytest

import torch

import kornia
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
        assert 0 <= ada_preset.p <= ada_preset.max_p # initial p
        assert ada_preset.real_acc_ema == .5
        assert ada_preset.num_calls == -ada_preset.update_every
        
        transforms = list(ada_preset.children())
        expected_transforms = [
            "RandomHorizontalFlip",
            "RandomRotation90",
            "RandomCrop",
            "RandomAffine",
            "ColorJitter",
            "RandomGaussianNoise"
        ]
        
        assert len(transforms) == len(expected_transforms)
        for t, et in zip(transforms, expected_transforms):
            assert et == str(t.__class__.__name__)

