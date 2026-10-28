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

from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomRotation90
from kornia.augmentation.presets.ada import AdaptiveDiscriminatorAugmentation

from testing.base import BaseTester


class PresetTests(BaseTester):
    pass


@pytest.mark.usefixtures("device", "dtype")
class TestAdaptiveDiscriminatorAugmentation(PresetTests):
    @pytest.mark.parametrize("seed", [0, 2])
    @pytest.mark.parametrize("direct_forward", [False, True])
    def test_shared_batch_gate(self, device, dtype, seed, direct_forward):
        inputs = torch.arange(4, device=device, dtype=dtype).reshape(1, 1, 1, 4).expand(8, 1, 1, 4).clone()
        expected_inputs = inputs.clone()
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=0.5, same_on_batch=True)
        torch.manual_seed(seed)
        actual = ada.forward(inputs) if direct_forward else ada(inputs)
        self.assert_close(actual, actual[:1].expand_as(actual), atol=0, rtol=0)
        assert torch.equal(actual, inputs) or torch.equal(actual, inputs.flip(-1))
        self.assert_close(inputs[0, 0, 0], torch.arange(4, device=device, dtype=dtype), atol=0, rtol=0)
        self.assert_close(inputs, expected_inputs, atol=0, rtol=0)

    @pytest.mark.parametrize("same_on_batch", [False, None])
    def test_independent_batch_gate(self, device, dtype, same_on_batch):
        inputs = torch.arange(4, device=device, dtype=dtype).reshape(1, 1, 1, 4).expand(8, 1, 1, 4).clone()
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=0.5, same_on_batch=same_on_batch)
        torch.manual_seed(0)
        selected = torch.bernoulli(torch.full((8,), 0.5, device=device)).bool()
        expected = torch.where(selected[:, None, None, None], inputs.flip(-1), inputs)
        torch.manual_seed(0)
        self.assert_close(ada(inputs), expected, atol=0, rtol=0)

    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_empty_batch_gate_consumes_no_draws(self, device, dtype, same_on_batch):
        inputs = torch.zeros(0, 1, 1, 4, device=device, dtype=dtype)
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=0.5, same_on_batch=same_on_batch)
        torch.manual_seed(0)
        expected = torch.rand(4, device=device)
        torch.manual_seed(0)
        self.assert_close(ada(inputs), inputs, atol=0, rtol=0)
        self.assert_close(torch.rand(4, device=device), expected, atol=0, rtol=0)

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

    @pytest.mark.parametrize("initial_p", [1e-5, 0.0, 0.5])
    def test_default_stack_raises_on_a_non_square_image_at_the_first_call(self, device, dtype, initial_p):
        ada = AdaptiveDiscriminatorAugmentation(initial_p=initial_p).to(device)
        inputs = torch.rand(8, 3, 16, 24, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="RandomRotation90"):
            ada(inputs)

    def test_non_square_check_reaches_a_nested_and_a_dict_input(self, device, dtype):
        nested = AugmentationSequential(RandomRotation90((1, 1), p=1.0))
        ada = AdaptiveDiscriminatorAugmentation(nested, initial_p=1e-5, data_keys=["input"])
        inputs = torch.rand(4, 3, 16, 24, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="RandomRotation90"):
            ada(inputs)
        with pytest.raises(ValueError, match="RandomRotation90"):
            ada({"input": inputs})

    @pytest.mark.parametrize("rotation", [RandomRotation90((2, 2), p=1.0), RandomHorizontalFlip(p=1.0)])
    def test_non_square_image_is_fine_when_no_rotation_may_turn_oddly(self, device, dtype, rotation):
        ada = AdaptiveDiscriminatorAugmentation(rotation, initial_p=0.5).to(device)
        inputs = torch.rand(4, 3, 16, 24, device=device, dtype=dtype)
        assert ada(inputs).shape == inputs.shape

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

    @pytest.mark.parametrize("route", ["call", "forward_data_keys", "init_data_keys"])
    def test_dict_input(self, device, dtype, route):
        # Every dictionary route raised: ADA always set `data_keys`, which the parent rejects with a dictionary, and
        # the default keys were the enum names ("IMAGE"), not the dictionary's. Each row is augmented with its mask,
        # and an entry that names no data key passes through.
        torch.manual_seed(0)
        x = torch.rand(16, 1, 4, 4, device=device, dtype=dtype)
        m = (torch.rand(16, 1, 4, 4, device=device) > 0.5).to(dtype)
        sample_id = torch.arange(16, device=device)
        data_keys = ["input", "mask"] if route == "init_data_keys" else None
        ada = AdaptiveDiscriminatorAugmentation(
            RandomHorizontalFlip(p=1.0), initial_p=0.5, max_p=1.0, data_keys=data_keys
        )
        inputs = {"input": x, "mask": m, "sample_id": sample_id}
        if route == "call":
            out = ada(inputs)
        elif route == "forward_data_keys":
            out = ada.forward(inputs, data_keys=["input", "mask"])
        else:
            out = ada.forward(inputs)

        assert set(out) == {"input", "mask", "sample_id"}
        flipped = (out["input"] != x).flatten(1).any(1)
        self.assert_close(out["input"][flipped], x[flipped].flip(-1))
        self.assert_close(out["input"][~flipped], x[~flipped])
        self.assert_close(out["mask"][flipped], m[flipped].flip(-1))
        self.assert_close(out["mask"][~flipped], m[~flipped])
        assert torch.equal(out["sample_id"], sample_id)

    def test_dict_input_data_keys_mismatch(self, device, dtype):
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=0.5)
        x = torch.rand(4, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="must match them, in any order"):
            ada.forward({"input": x, "mask": x}, data_keys=["input"])

    def test_dict_input_without_data_key(self, device, dtype):
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=0.5)
        x = torch.rand(4, 1, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="No key of the dictionary input names a data key"):
            ada({"img": x})

    def test_dict_input_data_keys_any_order(self, device, dtype):
        # The dictionary sets the order, so `data_keys` naming the same keys in another order are accepted.
        ada = AdaptiveDiscriminatorAugmentation(RandomHorizontalFlip(p=1.0), initial_p=1.0, max_p=1.0)
        x = torch.rand(4, 1, 4, 4, device=device, dtype=dtype)
        m = (x > 0.5).to(dtype)
        out = ada.forward({"mask": m, "input": x}, data_keys=["input", "mask"])
        self.assert_close(out["input"], x.flip(-1))
        self.assert_close(out["mask"], m.flip(-1))
