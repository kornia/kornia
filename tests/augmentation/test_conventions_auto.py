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

from testing.base import BaseTester, supports_bilinear_2d_grid_sample


class TestAutoAugmentConventions(BaseTester):
    def test_convention_policy_params_record_the_selected_path_and_replay(self, device, dtype):
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("bilinear 2D grid_sample is unavailable for this device and dtype")
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
    def test_convention_autoaugment_magnitude_bins_select_adjacent_intervals(self):
        degrees = AutoAugment(policy=[[("rotate", 1.0, 9)]]).forward_parameters(torch.Size([64, 1, 8, 6]))
        degrees = degrees[0].data[0].data["degrees"]
        assert (degrees >= 24.0).all() and (degrees <= 30.0).all()
        assert degrees.min() < 25.0 and degrees.max() > 29.0  # bounds alone pass for a collapsed interval
        torch.manual_seed(17)
        degrees = AutoAugment(policy=[[("rotate", 1.0, 5)]]).forward_parameters(torch.Size([64, 1, 8, 6]))
        degrees = degrees[0].data[0].data["degrees"]
        assert (degrees >= 0.0).all() and (degrees <= 6.0).all()  # bin 5 of linspace(-30, 30, 11) is [0, 6]
        assert degrees.min() < 1.0 and degrees.max() > 5.0
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
        # The drawn order is the execution order, not the policy-list order: 64 draws of 3 children reach more than
        # one of the six orders, and each parameter list replays its own order.
        orders = {tuple(param.name for param in aug.forward_parameters(torch.Size([2, 1, 8, 6]))) for _ in range(64)}
        assert len(orders) > 1
        image = torch.rand(2, 1, 8, 6)
        aug(image)
        drawn = [param.name for param in aug._params]
        assert [name for name, _ in aug.get_forward_sequence(aug._params)] == drawn

    @pytest.mark.device_agnostic
    def test_convention_trivialaugment_applies_the_symmetric_magnitude_mapping_4441(self):
        # #4441: TrivialAugment used to sample the wrapped augmentation directly, which skipped the random sign,
        # so a symmetric op such as rotate only ever drew non-negative magnitudes.
        torch.manual_seed(17)
        aug = TrivialAugment(policy=[[("rotate", -30.0, 30.0)]])
        degrees = aug.forward_parameters(torch.Size([64, 1, 8, 6]))[0].data[0].data["degrees"]
        assert (degrees.abs() <= 30.0).all()
        assert (degrees < 0).any() and (degrees > 0).any()
        assert degrees.abs().min() < 5.0 and degrees.abs().max() > 25.0

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
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("bilinear 2D grid_sample is unavailable for this device and dtype")
        image = torch.zeros(2, 1, 5, 5, device=device, dtype=dtype)
        image[..., 2, 2] = 1
        geometric = AutoAugment(policy=[[("translate_x", 1.0, 5)]])
        params = geometric.forward_parameters(image.shape)
        params[0].data[0].data["translate_x"].fill_(1.0)
        params[0].data[0].data["batch_prob"].fill_(1.0)
        output = geometric(image, params=params)
        expected_translation = torch.zeros_like(image)
        expected_translation[..., 2, 3] = 1
        self.assert_close(output, expected_translation)
        expected_translation_matrix = image.new_tensor([[1, 0, 1], [0, 1, 0], [0, 0, 1]]).expand(2, -1, -1)
        self.assert_close(geometric.transform_matrix, expected_translation_matrix)
        inverted = geometric.inverse(output, params=params)
        self.assert_close(inverted, image)

        ordered = AutoAugment(policy=[[("rotate", 1.0, 5), ("translate_x", 1.0, 5)]])
        marker = torch.zeros(2, 1, 5, 5, device=device, dtype=dtype)
        marker[..., 1, 2] = 1
        ordered_params = ordered.forward_parameters(marker.shape)
        ordered_params[0].data[0].data["degrees"].fill_(90.0)
        ordered_params[0].data[1].data["translate_x"].fill_(1.0)
        for item in ordered_params[0].data:
            item.data["batch_prob"].fill_(1.0)
        ordered_output = ordered(marker, params=ordered_params)
        expected_ordered_output = torch.zeros_like(marker)
        expected_ordered_output[..., 2, 2] = 1
        self.assert_close(ordered_output, expected_ordered_output)
        # Rotation about (2, 2), followed by translation: T @ R, not R @ T.
        expected_matrix = marker.new_tensor([[0, 1, 1], [-1, 0, 4], [0, 0, 1]]).expand(2, -1, -1)
        self.assert_close(ordered.transform_matrix, expected_matrix)
        recomputed = ordered[0].get_transformation_matrix(marker, params=ordered_params[0].data, recompute=True)
        self.assert_close(recomputed, expected_matrix)
        # The inverse undoes the translation before the rotation. Undoing them in forward order would leave the
        # marker, which sits on the rotation centre here, at (2, 1) instead.
        self.assert_close(ordered.inverse(ordered_output, params=ordered_params), marker)

        chained = RandAugment(n=2, m=15, policy=[[("rotate", -30.0, 30.0)], [("translate_x", -0.5, 0.5)]])
        chained_params = chained.forward_parameters(marker.shape)
        for item in chained_params:
            data = item.data[0].data
            data["batch_prob"].fill_(1.0)
            if "degrees" in data:
                data["degrees"].fill_(90.0)
            else:
                data["translate_x"].fill_(1.0)
        for execution_params in (chained_params, list(reversed(chained_params))):
            rotate_first = "degrees" in execution_params[0].data[0].data
            chained_output = chained(marker, params=execution_params)
            # The supplied parameter order selects T @ R, or R @ T when the translation leads.
            chained_matrix = expected_matrix if rotate_first else marker.new_tensor([[0, 1, 0], [-1, 0, 3], [0, 0, 1]])
            self.assert_close(chained.transform_matrix, chained_matrix.expand(2, -1, -1))
            recomputed = chained.get_transformation_matrix(marker, params=execution_params, recompute=True)
            self.assert_close(recomputed, chained_matrix.expand(2, -1, -1))
            expected_chained_output = torch.zeros_like(marker)
            if rotate_first:
                expected_chained_output[..., 2, 2] = 1
            else:
                expected_chained_output[..., 1, 1] = 1
            self.assert_close(chained_output, expected_chained_output)

        intensity = AutoAugment(policy=[[("solarize", 1.0, 5)]])
        intensity(image)
        self.assert_close(intensity.transform_matrix, torch.eye(3, device=device, dtype=dtype).expand(2, -1, -1))
        with pytest.raises(RuntimeError, match="is not supported"):
            intensity.inverse(image)

        skipped_intensity = AutoAugment(policy=[[("solarize", 1.0, 5)]])
        skipped_params = skipped_intensity.forward_parameters(image.shape)
        skipped_params[0].data[0].data["batch_prob"].zero_()
        skipped_output = skipped_intensity(image, params=skipped_params)
        self.assert_close(skipped_output, image)
        self.assert_close(skipped_intensity.inverse(skipped_output, params=skipped_params), image)
        # One applied row is enough to make the path non-invertible.
        skipped_params[0].data[0].data["batch_prob"][0] = 1.0
        partial_output = skipped_intensity(image, params=skipped_params)
        with pytest.raises(RuntimeError, match="is not supported"):
            skipped_intensity.inverse(partial_output, params=skipped_params)

    @pytest.mark.device_agnostic
    def test_convention_policy_sequential_supplied_params_define_the_execution_path(self):
        policy = PolicySequential(
            ops.Invert(initial_probability=1.0), ops.Solarize(initial_magnitude=0.5, initial_probability=1.0)
        )
        image = torch.tensor([0.2], dtype=torch.float32).reshape(1, 1, 1, 1)
        generated = policy.forward_parameters(image.shape)
        for param in generated:
            param.data["batch_prob"].fill_(1.0)
            if "thresholds" in param.data:
                param.data["thresholds"].fill_(0.5)
                param.data["additions"].zero_()
        self.assert_close(policy(image, params=generated), torch.tensor([0.2]).reshape_as(image))
        reversed_params = list(reversed(generated))
        self.assert_close(policy(image, params=reversed_params), torch.tensor([0.8]).reshape_as(image))
        self.assert_close(policy(image, params=generated[:1]), torch.tensor([0.8]).reshape_as(image))
        assert [param.name for param in policy._params] == [generated[0].name]
        # get_forward_sequence, which get_transformation_matrix zips against, follows the supplied order too.
        reversed_names = [name for name, _ in policy.get_forward_sequence(reversed_params)]
        assert reversed_names == [param.name for param in reversed_params]
        assert [name for name, _ in policy.get_forward_sequence(generated[:1])] == [generated[0].name]

    @pytest.mark.device_agnostic
    def test_convention_intensity_matrix_requires_a_nonempty_selected_policy(self):
        image = torch.rand(1, 1, 3, 3)
        intensity = AutoAugment(policy=[[("solarize", 1.0, 5)]])
        intensity(image)
        self.assert_close(intensity.transform_matrix, torch.eye(3).unsqueeze(0))
        empty = AutoAugment(policy=[[]])
        self.assert_close(empty(image), image)
        assert empty.transform_matrix is None

    @pytest.mark.device_agnostic
    def test_convention_symmetric_magnitude_preserves_the_post_mapping_value(self):
        posterize = ops.Posterize(initial_magnitude=0.5, magnitude_range=(0.0, 8.0), symmetric_megnitude=True)
        bits = posterize.forward_parameters(torch.Size([32, 1, 1, 1]))["bits_factor"]
        self.assert_close(bits, torch.zeros_like(bits), rtol=0, atol=0)
        # A zero is sign-blind, so it cannot show that the mapping runs BEFORE the sign is chosen.
        nonzero = ops.Posterize(initial_magnitude=5.5, magnitude_range=(0.0, 8.0), symmetric_megnitude=True)
        signed = nonzero.forward_parameters(torch.Size([64, 1, 1, 1]))["bits_factor"]
        assert set(signed.tolist()) == {5, -5}
        # Every built-in mapping is odd (identity, x * 180, truncation), so {5, -5} holds for either order. A
        # mapping that is not odd separates them: sign after gives {13, -13}, sign before would give {13, 7}.
        from kornia.augmentation import RandomRotation
        from kornia.augmentation.auto.operations.base import OperationBase

        shifted = OperationBase(
            RandomRotation((0.0, 30.0), p=1.0),
            initial_magnitude=[("degrees", 3.0)],
            magnitude_fn=lambda magnitude: magnitude + 10.0,
            symmetric_megnitude=True,
        )
        degrees = shifted.forward_parameters(torch.Size([64, 1, 4, 4]))["degrees"]
        assert set(degrees.tolist()) == {13.0, -13.0}

    @pytest.mark.device_agnostic
    def test_convention_operation_probability_and_magnitude_clamps(self):
        initial = ops.Invert(initial_probability=0.25)
        self.assert_close(initial.probability.detach(), torch.tensor([0.25]), rtol=0, atol=0)
        operation = ops.Rotate(initial_magnitude=3.0, initial_probability=0.5)
        operation._probability.data.fill_(2.0)
        operation._magnitude.data.fill_(100.0)
        self.assert_close(operation.probability, torch.full_like(operation.probability, 1.0 - 1e-7), rtol=0, atol=0)
        self.assert_close(operation.magnitude, torch.full_like(operation.magnitude, 30.0))
        operation._probability.data.fill_(-1.0)
        self.assert_close(operation.probability, torch.full_like(operation.probability, 1e-7), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "probability,expected",
        [(1.0, [0.375, 0.5, 0.375, 0.25]), (0.5, [0.25, 0.25, 0.375, 0.25])],
    )
    def test_convention_operation_soft_blend_respects_the_wrapped_gate(self, probability, expected, device, dtype):
        # The wrapped p=1 path transforms every row. At p<1 it first keeps gates <=0.5 unchanged,
        # so the outer blend cannot mix those rows with the transformed image.
        invert = ops.Invert(initial_probability=probability)
        image = torch.tensor([0.25, 0.25, 0.75, 0.75], device=device, dtype=dtype).view(4, 1, 1, 1)
        params = invert.op.forward_parameters(image.shape)
        params["batch_prob"] = torch.tensor([0.25, 0.5, 0.75, 1.0])
        self.assert_close(invert(image, params=params).flatten(), image.new_tensor(expected))

    @pytest.mark.device_agnostic
    def test_convention_policy_sequential_samples_through_the_operation_wrapper_4441(self):
        # #4441: PolicySequential used to call the wrapped augmentation's forward_parameters, so it ignored the
        # wrapper's magnitude and magnitude mapping. The gate still comes from the wrapped augmentation's p, and
        # the wrapper's probability parameter is still not consulted on either path.
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
        assert direct_params.keys() == wrapped_params.keys()
        for key, value in wrapped_params.items():
            self.assert_close(direct_params[key], value)
        self.assert_close(operation.probability, torch.tensor([1e-7]))
        assert direct_operation.op.p == 0.5

    def test_convention_recorded_params_reproduce_the_sampled_forward(self, device, dtype):
        if not supports_bilinear_2d_grid_sample(device, dtype):
            pytest.skip("bilinear 2D grid_sample is unavailable for this device and dtype")
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
        # A range that does not start at zero separates ``low + (high - low) * m / 30`` from ``high * m / 30``.
        aug = RandAugment(n=1, m=6, policy=[[("brightness", 0.5, 1.5)]])
        factor = aug.forward_parameters(torch.Size([8, 3, 8, 6]))[0].data[0].data["brightness_factor"]
        self.assert_close(factor, torch.full_like(factor, 0.7))

    @pytest.mark.device_agnostic
    def test_convention_autoaugment_posterize_rounds_its_interval(self):
        torch.manual_seed(17)
        aug = AutoAugment(policy=[[("posterize", 1.0, 1)]])
        bits = aug.forward_parameters(torch.Size([256, 3, 8, 8]))[0].data[0].data["bits_factor"]
        assert not bits.is_floating_point()
        assert set(bits.tolist()) == {4, 5}  # bin 1 is the interval (4.4, 4.8)
        # Bin 2 is entirely above 4.5 and below 5.5, so every draw rounds to 5; truncation would return 4 or 5.
        aug = AutoAugment(policy=[[("posterize", 1.0, 2)]])
        bits = aug.forward_parameters(torch.Size([256, 3, 8, 8]))[0].data[0].data["bits_factor"]
        self.assert_close(bits, torch.full_like(bits, 5), rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_randaugment_posterize_and_translate_units_4655(self):
        from kornia.enhance import posterize as posterize_bits

        image = torch.rand(4, 3, 20, 32)
        # The range runs backwards, high - (high - low) * m / 30, and is then truncated: a larger m keeps fewer bits.
        for m, bits in ((7, 7), (15, 6), (29, 4)):
            posterize = RandAugment(n=1, m=m, policy=[[("posterize", 4.0, 8.0)]])
            output = posterize(image)
            assert posterize._params[0].data[0].data["bits_factor"].tolist() == [bits] * 4
            self.assert_close(output, posterize_bits(image, bits), rtol=0, atol=0)
        # A translate range is a fraction of the width (x) or the height (y); H != W tells the two apart.
        for name, pixels in (("translate_x", 0.05 * 32), ("translate_y", 0.05 * 20)):
            translate = RandAugment(n=1, m=15, policy=[[(name, -0.1, 0.1)]])
            drawn = translate.forward_parameters(image.shape)[0].data[0].data[name]
            self.assert_close(drawn.abs(), torch.full_like(drawn, pixels))

    @pytest.mark.device_agnostic
    def test_convention_operation_probability_parameter_is_inert_4656(self):
        import copy

        gates = []
        for learned in (1e-7, 1.0 - 1e-7):
            operation = ops.Invert(initial_probability=0.5)
            operation._probability.data.fill_(learned)
            torch.manual_seed(17)
            batch_prob = operation.forward_parameters(torch.Size([4000, 1, 4, 4]))["batch_prob"]
            assert set(batch_prob.unique().tolist()) == {0.0, 1.0}
            assert 1700 < batch_prob.sum() < 2300
            assert not batch_prob.requires_grad
            gates.append(batch_prob)
        self.assert_close(gates[0], gates[1], rtol=0, atol=0)
        operation = ops.Brightness(initial_magnitude=0.3, initial_probability=0.5)
        operation(torch.rand(8, 3, 4, 4)).sum().backward()
        assert operation._probability.grad is None
        assert operation._magnitude.grad is not None
        # The parameter is kept in the state dict, and no sampler is stored on the wrapper or the wrapped op.
        assert "_probability" in operation.state_dict()
        assert not hasattr(operation.op, "_p_gen") and not hasattr(operation.op, "_p_batch_gen")
        # So a policy deep-copies after a forward and after train() / eval(), and the copy replays the original.
        image = torch.rand(2, 3, 8, 8)
        for policy in (AutoAugment(), TrivialAugment(), RandAugment(n=2, m=15)):
            copy.deepcopy(policy.eval())
            copy.deepcopy(policy.train())
            output = policy(image)
            self.assert_close(copy.deepcopy(policy)(image, params=policy._params), output, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    def test_convention_trivialaugment_fixes_candidate_probability_at_one(self):
        aug = TrivialAugment(policy=[[("rotate", -30.0, 30.0)], [("brightness", 0.1, 1.9)]])
        assert {(module[0].op.p, module[0].op.p_batch) for _, module in aug.named_children()} == {(1.0, 1.0)}
        # "Each candidate": every entry of the default policy as well, not only the two above.
        defaults = [(module[0].op.p, module[0].op.p_batch) for _, module in TrivialAugment().named_children()]
        assert len(defaults) == 12 and set(defaults) == {(1.0, 1.0)}
        gates = set()
        for _ in range(16):
            gates.update(aug.forward_parameters(torch.Size([8, 3, 8, 6]))[0].data[0].data["batch_prob"].tolist())
        assert gates == {1.0}
        # Only the magnitude is drawn per row.
        torch.manual_seed(17)
        magnitudes = (
            TrivialAugment(policy=[[("rotate", -30.0, 30.0)]])
            .forward_parameters(torch.Size([8, 1, 8, 6]))[0]
            .data[0]
            .data["degrees"]
        )
        assert magnitudes.unique().numel() == 8

    @pytest.mark.device_agnostic
    def test_convention_policy_shear_entries_are_mapped_to_degrees_4441(self):
        # #4441: TrivialAugment used to drop ShearX's 180 factor, so ("shear_x", -0.3, 0.3) sheared by at most
        # 0.3 degrees, where RandAugment sheared by up to 54 degrees for the same entry.
        torch.manual_seed(17)
        trivial = (
            TrivialAugment(policy=[[("shear_x", -0.3, 0.3)]])
            .forward_parameters(torch.Size([64, 1, 8, 6]))[0]
            .data[0]
            .data["shear_x"]
        )
        assert bool((trivial.abs() <= 0.3 * 180).all()) and trivial.abs().max() > 30.0
        assert (trivial < 0).any() and (trivial > 0).any()
        # AutoAugment's shear bins are fractions too, so the mapping is applied once: bin b of either shear op spans
        # the adjacent points b and b + 1 of linspace(-0.3, 0.3, 11), times 180, and 256 rows reach both ends.
        edges = [-0.3 + 0.06 * point for point in range(11)]
        for name in ("shear_x", "shear_y"):
            for magnitude_bin in range(10):
                low, high = edges[magnitude_bin] * 180, edges[magnitude_bin + 1] * 180
                auto = (
                    AutoAugment(policy=[[(name, 1.0, magnitude_bin)]])
                    .forward_parameters(torch.Size([256, 1, 8, 6]))[0]
                    .data[0]
                    .data[name]
                )
                assert bool((auto >= low - 1e-3).all()) and bool((auto <= high + 1e-3).all()), (name, magnitude_bin)
                assert auto.min() < low + 1.0 and auto.max() > high - 1.0, (name, magnitude_bin)
        # The same policy entry through RandAugment, which always applied the mapping.
        mapped = (
            RandAugment(n=1, m=29, policy=[[("shear_x", -0.3, 0.3)]])
            .forward_parameters(torch.Size([8, 1, 8, 6]))[0]
            .data[0]
            .data["shear_x"]
        )
        self.assert_close(mapped.abs(), torch.full_like(mapped, 0.29 * 180))

    @pytest.mark.device_agnostic
    def test_wart_operation_wrappers_cannot_be_pickled_4469(self):
        import pickle

        # Python 3.14 raises PicklingError for a local object where earlier versions raise AttributeError.
        local_object_error = (AttributeError, pickle.PicklingError)
        failed = set()
        for name in ops.__all__:
            operation = getattr(ops, name)()
            pickle.loads(pickle.dumps(operation.op))  # noqa: S301 - the wrapped augmentation always pickles
            try:
                pickle.loads(pickle.dumps(operation))  # noqa: S301
            except local_object_error:
                failed.add(name)
        # With default arguments only Posterize avoids a local closure: it passes a named mapping and no sign flip.
        assert failed == set(ops.__all__) - {"Posterize"}
        # It is the configuration, not the class: ShearX / ShearY pass a named mapping too, and pickle without
        # the sign flip, while the sign flip makes Posterize unpicklable.
        for operation in (ops.ShearX(symmetric_megnitude=False), ops.ShearY(symmetric_megnitude=False)):
            pickle.loads(pickle.dumps(operation))  # noqa: S301
        with pytest.raises(local_object_error, match="local object"):
            pickle.dumps(ops.Posterize(symmetric_megnitude=True))
        with pytest.raises(local_object_error, match="local object"):
            pickle.dumps(ops.Rotate(symmetric_megnitude=False))  # no named mapping: the identity closure
        # So the default policies, which hold such wrappers, do not pickle either.
        for policy in (AutoAugment(), RandAugment(n=2, m=15), TrivialAugment()):
            with pytest.raises(local_object_error, match="local object"):
                pickle.dumps(policy)

    @pytest.mark.device_agnostic
    def test_convention_rigid_matrix_mode_accepts_intensity_operations(self):
        # "rigid" rejects non-rigid modules in other containers; a policy has none, since every wrapper has a matrix.
        image = torch.rand(2, 3, 8, 6)
        for mode in ("silence", "rigid"):
            intensity = AutoAugment(policy=[[("invert", 1.0, None)]], transformation_matrix_mode=mode)
            intensity(image)
            self.assert_close(intensity.transform_matrix, torch.eye(3).expand(2, 3, 3))
            mixed = AutoAugment(policy=[[("rotate", 1.0, 9), ("invert", 1.0, None)]], transformation_matrix_mode=mode)
            mixed(image)
            assert not torch.allclose(mixed.transform_matrix, torch.eye(3).expand(2, 3, 3))

    @pytest.mark.device_agnostic
    def test_convention_policy_sequential_takes_operations_positionally(self):
        operations = [ops.Invert(initial_probability=1.0), ops.Solarize(initial_magnitude=0.5, initial_probability=1.0)]
        assert len(PolicySequential(*operations)) == 2
        with pytest.raises(ValueError, match="must be Kornia Operations"):
            PolicySequential(operations)

    @pytest.mark.device_agnostic
    def test_convention_randaugment_formula_holds_for_every_default_entry(self):
        # "The identity for every default entry except three": the whole default policy, not a sample of it.
        import math

        from kornia.augmentation.auto.rand_augment.rand_augment import default_policy

        checked = []
        height, width = 8, 6
        for subpolicy in default_policy:
            name = subpolicy[0][0]
            aug = RandAugment(n=1, m=10, policy=[subpolicy])
            operation = aug[0][0]
            if operation._factor_name is None:
                continue  # auto_contrast, equalize and invert carry no magnitude
            low, high = (float(bound) for bound in operation.magnitude_range)
            expected = low + (high - low) * 10 / 30
            if name in ("shear_x", "shear_y"):
                expected *= 180
            elif name == "posterize":
                # The range runs backwards: 8 - 4 * 10 / 30 = 6.67 -> 6 bits, truncation and not rounding.
                expected = math.floor(high - (high - low) * 10 / 30)
            elif name in ("translate_x", "translate_y"):
                expected *= width if name == "translate_x" else height  # a fraction of that side, in pixels
            drawn = aug.forward_parameters(torch.Size([8, 1, height, width]))[0].data[0].data[operation._factor_name]
            self.assert_close(drawn.abs().float(), torch.full((8,), float(expected)))
            checked.append(name)
        assert len(checked) == 12 and {"shear_x", "shear_y", "posterize", "translate_x", "translate_y"} <= set(checked)
