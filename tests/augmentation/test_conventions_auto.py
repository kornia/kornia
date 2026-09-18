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

from __future__ import annotations

import pytest
import torch

from kornia.augmentation.auto.autoaugment import AutoAugment
from kornia.augmentation.auto.operations import PolicySequential, ops
from kornia.augmentation.auto.rand_augment import RandAugment
from kornia.augmentation.auto.trivial_augment import TrivialAugment

from testing.base import BaseTester


class TestAutoAugmentConventions(BaseTester):
    def test_convention_policy_params_record_the_selected_path_and_replay(self, device, dtype):
        image = torch.rand(3, 1, 8, 6, device=device, dtype=dtype)
        cases = [
            (AutoAugment(policy=[[("rotate", 1.0, 5)]]), 1),
            (RandAugment(n=2, m=15, policy=[[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)]]), 2),
            (TrivialAugment(policy=[[("rotate", -30.0, 30.0)]]), 1),
        ]
        for aug, expected_operations in cases:
            params = aug.forward_parameters(image.shape)
            assert len(params) == expected_operations
            assert all(len(param.data) == 1 for param in params)
            output = aug(image, params=params)
            self.assert_close(output, aug(image, params=params))

    @pytest.mark.device_agnostic
    def test_convention_autoaugment_and_trivialaugment_choose_one_policy(self):
        torch.manual_seed(17)
        auto_policy = [[("invert", 1.0, None)], [("solarize", 1.0, 5)]]
        trivial_policy = [[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)]]
        for aug in (AutoAugment(policy=auto_policy), TrivialAugment(policy=trivial_policy)):
            selected = {aug.forward_parameters(torch.Size([2, 1, 8, 6]))[0].name for _ in range(32)}
            assert selected == set(dict(aug.named_children()))
            self.assert_close(aug.rand_selector.probs, torch.full((2,), 0.5))

    @pytest.mark.device_agnostic
    def test_convention_recorded_params_select_the_recorded_children(self):
        policy = [[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)], [("translate_y", -0.5, 0.5)]]
        auto_policy = [[("rotate", 1.0, 5)], [("translate_x", 1.0, 5)], [("translate_y", 1.0, 5)]]
        for aug in (
            AutoAugment(policy=auto_policy),
            RandAugment(n=2, m=15, policy=policy),
            TrivialAugment(policy=policy),
        ):
            params = aug.forward_parameters(torch.Size([2, 1, 8, 6]))
            for _ in range(16):
                assert [name for name, _ in aug.get_forward_sequence(params)] == [param.name for param in params]

    @pytest.mark.device_agnostic
    def test_convention_autoaugment_magnitude_bins_are_zero_through_nine(self):
        degrees = AutoAugment(policy=[[("rotate", 1.0, 9)]]).forward_parameters(torch.Size([64, 1, 8, 6]))
        degrees = degrees[0].data[0].data["degrees"]
        assert (degrees >= 24.0).all() and (degrees <= 30.0).all()
        for magnitude in (-1, 10):
            with pytest.raises(ValueError, match=r"in \[0, 9\]"):
                AutoAugment(policy=[[("rotate", 1.0, magnitude)]])

    @pytest.mark.device_agnostic
    def test_convention_randaugment_selects_distinct_candidates(self):
        policy = [
            [("rotate", -30.0, 30.0)],
            [("translate_x", -0.5, 0.5)],
            [("translate_y", -0.5, 0.5)],
        ]
        aug = RandAugment(n=3, m=15, policy=policy)
        for _ in range(8):
            params = aug.forward_parameters(torch.Size([2, 1, 8, 6]))
            assert len({param.name for param in params}) == 3

    @pytest.mark.device_agnostic
    def test_convention_autoaugment_magnitude_bin_samples_its_adjacent_interval(self):
        torch.manual_seed(17)
        aug = AutoAugment(policy=[[("rotate", 1.0, 5)]])
        degrees = aug.forward_parameters(torch.Size([64, 1, 8, 6]))[0].data[0].data["degrees"]
        assert (degrees >= 0.0).all()
        assert (degrees <= 6.0).all()
        assert degrees.min() < 1.0 and degrees.max() > 5.0

    @pytest.mark.device_agnostic
    def test_wart_trivialaugment_bypasses_symmetric_magnitude_mapping_4441(self):
        torch.manual_seed(17)
        aug = TrivialAugment(policy=[[("rotate", -30.0, 30.0)]])
        degrees = aug.forward_parameters(torch.Size([64, 1, 8, 6]))[0].data[0].data["degrees"]
        assert (degrees >= 0.0).all()
        assert (degrees <= 30.0).all()
        assert degrees.min() < 5.0 and degrees.max() > 25.0

    @pytest.mark.device_agnostic
    def test_convention_randaugment_maps_m_and_validates_the_policy_cardinality(self):
        policy = [[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)]]
        torch.manual_seed(17)
        aug = RandAugment(n=2, m=15, policy=policy)
        params = aug.forward_parameters(torch.Size([64, 1, 8, 6]))
        degrees = next(param for param in params if "degrees" in param.data[0].data).data[0].data["degrees"]
        self.assert_close(degrees.abs(), torch.full_like(degrees, 15.0))
        assert (degrees < 0).any() and (degrees > 0).any()
        for m in (0, 30):
            with pytest.raises(ValueError, match=r"Expect `m` in \(0, 30\)"):
                RandAugment(n=1, m=m, policy=policy)
        for n in (0, 3):
            with pytest.raises(ValueError, match=r"Expect `n` in \[1, 2\]"):
                RandAugment(n=n, m=15, policy=policy)

    def test_convention_policy_matrix_composes_and_inverse_refuses_intensity(self, device, dtype):
        image = torch.rand(2, 1, 8, 6, device=device, dtype=dtype)
        geometric = AutoAugment(policy=[[("translate_x", 1.0, 5)]])
        params = geometric.forward_parameters(image.shape)
        params[0].data[0].data["translate_x"].zero_()
        output = geometric(image, params=params)
        self.assert_close(geometric.transform_matrix, torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1))
        inverted = geometric.inverse(output, params=params)
        assert inverted.shape == image.shape
        if dtype not in (torch.float16, torch.bfloat16):
            self.assert_close(inverted, image)

        ordered = AutoAugment(policy=[[("rotate", 1.0, 5), ("translate_x", 1.0, 5)]])
        marker = torch.zeros(2, 1, 5, 5, device=device, dtype=dtype)
        marker[..., 2, 2] = 1
        ordered_params = ordered.forward_parameters(marker.shape)
        ordered_params[0].data[0].data["degrees"].fill_(90.0)
        ordered_params[0].data[1].data["translate_x"].fill_(1.0)
        for item in ordered_params[0].data:
            item.data["batch_prob"].fill_(1.0)
        ordered(marker, params=ordered_params)
        # Rotation about (2, 2), followed by translation: T @ R, not R @ T.
        expected_matrix = marker.new_tensor([[0, 1, 1], [-1, 0, 4], [0, 0, 1]]).expand(2, -1, -1)
        self.assert_close(ordered.transform_matrix, expected_matrix)
        recomputed = ordered[0].get_transformation_matrix(marker, params=ordered_params[0].data, recompute=True)
        self.assert_close(recomputed, expected_matrix)

        chained = RandAugment(n=2, m=15, policy=[[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)]])
        chained_params = chained.forward_parameters(marker.shape)
        rotate_first = "degrees" in chained_params[0].data[0].data
        for item in chained_params:
            data = item.data[0].data
            data["batch_prob"].fill_(1.0)
            if "degrees" in data:
                data["degrees"].fill_(90.0)
            else:
                data["translate_x"].fill_(1.0)
        chained(marker, params=chained_params)
        # Whichever is drawn first is applied first: T @ R, or R @ T when the translation leads.
        chained_matrix = expected_matrix if rotate_first else marker.new_tensor([[0, 1, 0], [-1, 0, 3], [0, 0, 1]])
        self.assert_close(chained.transform_matrix, chained_matrix.expand(2, -1, -1))
        recomputed = chained.get_transformation_matrix(marker, params=chained_params, recompute=True)
        self.assert_close(recomputed, chained_matrix.expand(2, -1, -1))

        intensity = AutoAugment(policy=[[("solarize", 1.0, 5)]])
        intensity(image)
        self.assert_close(intensity.transform_matrix, torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1))
        with pytest.raises(RuntimeError, match="is not supported"):
            intensity.inverse(image)

    @pytest.mark.device_agnostic
    def test_convention_operation_probability_magnitude_and_soft_blend(self):
        operation = ops.Rotate(initial_magnitude=3.0, initial_probability=0.5)
        operation._probability.data.fill_(2.0)
        operation._magnitude.data.fill_(100.0)
        self.assert_close(operation.probability, torch.full_like(operation.probability, 1.0 - 1e-7), rtol=0, atol=0)
        self.assert_close(operation.magnitude, torch.full_like(operation.magnitude, 30.0))
        operation._probability.data.fill_(-1.0)
        self.assert_close(operation.probability, torch.full_like(operation.probability, 1e-7), rtol=0, atol=0)

        invert = ops.Invert(initial_probability=1.0)
        image = torch.tensor([[[[0.2]]], [[[0.8]]]])
        params = invert.op.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([0.25, 0.75])
        batch_prob = params["batch_prob"][:, None, None, None]
        expected = batch_prob * (1.0 - image) + (1.0 - batch_prob) * image
        self.assert_close(invert(image, params=params), expected)

    @pytest.mark.device_agnostic
    def test_wart_policy_sequential_bypasses_operation_wrapper_sampling_4441(self):
        operation = ops.Rotate(initial_magnitude=3.0, initial_probability=0.5)
        direct_operation = ops.Rotate(initial_magnitude=3.0, initial_probability=0.5)
        direct_policy = PolicySequential(direct_operation)
        shape = torch.Size([4, 1, 8, 6])

        operation._probability.data.fill_(1e-7)
        direct_operation._probability.data.fill_(1e-7)
        torch.manual_seed(3)
        wrapped_params = operation.forward_parameters(shape)
        torch.manual_seed(3)
        direct_params = direct_policy.forward_parameters(shape)[0].data

        self.assert_close(wrapped_params["degrees"].abs(), torch.full_like(wrapped_params["degrees"], 3.0))
        assert not torch.equal(direct_params["degrees"], wrapped_params["degrees"])
        assert not torch.allclose(direct_params["degrees"].abs(), torch.full_like(direct_params["degrees"], 3.0))
        self.assert_close(operation.probability, torch.tensor([1e-7]))
        assert direct_operation.op.p == 0.5

    def test_convention_recorded_params_reproduce_the_sampled_forward(self, device, dtype):
        image = torch.rand(3, 3, 8, 8, device=device, dtype=dtype)
        for aug in (AutoAugment(), RandAugment(n=2, m=15), TrivialAugment()):
            for seed in range(4):
                torch.manual_seed(seed)
                output = aug(image)
                self.assert_close(aug(image, params=aug._params), output, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_randaugment_magnitude_formula(self):
        aug = RandAugment(n=1, m=3, policy=[[("rotate", -30.0, 30.0)]])
        degrees = aug.forward_parameters(torch.Size([64, 1, 8, 6]))[0].data[0].data["degrees"]
        self.assert_close(degrees.abs(), torch.full_like(degrees, 3.0))

    @pytest.mark.device_agnostic
    def test_convention_autoaugment_posterize_truncates_its_interval(self):
        aug = AutoAugment(policy=[[("posterize", 1.0, 1)]])
        bits = aug.forward_parameters(torch.Size([256, 3, 8, 8]))[0].data[0].data["bits_factor"]
        assert not bits.is_floating_point()
        assert set(bits.tolist()) == {4, 5}  # bin 1 is the interval (4.4, 4.8)

    @pytest.mark.device_agnostic
    def test_wart_randaugment_posterize_and_translate_units_4655(self):
        image = torch.rand(4, 3, 32, 32)
        posterize = RandAugment(n=1, m=7, policy=[[("posterize", 0.0, 4)]])
        output = posterize(image)
        assert posterize._params[0].data[0].data["bits_factor"].tolist() == [0, 0, 0, 0]
        self.assert_close(output, torch.zeros_like(image))
        translate = RandAugment(n=1, m=29, policy=[[("translate_x", -0.5, 0.5)]])
        pixels = translate.forward_parameters(image.shape)[0].data[0].data["translate_x"]
        self.assert_close(pixels.abs(), torch.full_like(pixels, 0.5 * 29 / 30))

    @pytest.mark.device_agnostic
    def test_wart_operation_probability_parameter_is_inert_4656(self):
        import copy

        for learned in (1e-7, 1.0 - 1e-7):
            operation = ops.Invert(initial_probability=0.5)
            operation._probability.data.fill_(learned)
            batch_prob = operation.forward_parameters(torch.Size([4000, 1, 4, 4]))["batch_prob"]
            assert set(batch_prob.unique().tolist()) == {0.0, 1.0}
            assert 1700 < batch_prob.sum() < 2300
        operation = ops.Brightness(initial_magnitude=0.3, initial_probability=0.5)
        operation(torch.rand(8, 3, 4, 4)).sum().backward()
        assert operation._probability.grad is None
        assert operation._magnitude.grad is not None
        policy = RandAugment(n=2, m=15)
        copy.deepcopy(policy)
        policy(torch.rand(2, 3, 8, 8))
        with pytest.raises(RuntimeError, match="graph leaves"):
            copy.deepcopy(policy)
