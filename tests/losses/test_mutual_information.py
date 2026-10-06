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

from kornia.losses.mutual_information import (
    MIKernel,
    MILossFromRef,
    NMILossFromRef,
    _normalize_signal,
    mutual_information_loss,
    mutual_information_loss_2d,
    normalized_mutual_information_loss,
    normalized_mutual_information_loss_2d,
)

from testing.base import BaseTester


class TestMutualInformationLoss(BaseTester):
    @staticmethod
    def relative_mi(img_1, img_2, window_radius):
        """Should theoretically be 0 if img_1 and img_2 are independent and 1 if img_1 = f(img_2), f one to one."""
        numerator = mutual_information_loss(img_1, img_2, window_radius=window_radius)
        denominator = mutual_information_loss(img_2, img_2, window_radius=window_radius)
        return numerator / denominator

    @staticmethod
    def sampling_function(n_samples, device, dtype):
        data = torch.rand(n_samples, device=device, dtype=dtype)
        return 400 * torch.sin(data * torch.pi)

    def value_ranges_check(self, device, dtype, n_samples=10000, num_bins=64):
        img_1 = self.sampling_function(n_samples, device, dtype)
        img_2 = 50 * img_1 + 1
        img_3 = self.sampling_function(n_samples, device, dtype)

        for radius in [1 / 2, 1, 2, 3]:
            # relative MI, expect 1
            assert torch.allclose(
                self.relative_mi(img_1, img_2, window_radius=radius), torch.ones(1, dtype=dtype, device=device)
            ), "Wrong MI behaviour, correlated case."
            # relative MI, expect 0
            # NOTE: mutual_information_loss is a finite-sample, histogram-based estimator applied to random data.
            # For independent variables the theoretical value is 0, but sampling noise and binning effects across
            # radii introduce noticeable variance, so we use a slightly looser atol here for test robustness.
            assert torch.allclose(
                self.relative_mi(img_1, img_3, window_radius=radius),
                torch.zeros(1, dtype=dtype, device=device),
                atol=0.2,
            ), "Wrong MI behaviour, uncorrelated case."

            assert torch.allclose(
                self.relative_mi(img_2, img_3, window_radius=radius),
                torch.zeros(1, dtype=dtype, device=device),
                atol=0.2,
            ), "Wrong MI behaviour, uncorrelated case."

            # NMI, expect -2
            assert torch.allclose(
                normalized_mutual_information_loss(img_1, img_2, window_radius=radius, num_bins=num_bins),
                -2 * torch.ones(1, dtype=dtype, device=device),
                atol=0.2 * radius + 0.15,
            ), "Wrong NMI behaviour, correlated case."

            # NMI, expect -1
            assert torch.allclose(
                normalized_mutual_information_loss(img_1, img_3, window_radius=radius, num_bins=num_bins),
                -torch.ones(1, dtype=dtype, device=device),
                atol=0.1,
            ), "Wrong NMI behaviour, uncorrelated case."
            assert torch.allclose(
                normalized_mutual_information_loss(img_2, img_3, window_radius=radius, num_bins=num_bins),
                -torch.ones(1, dtype=dtype, device=device),
                atol=0.1,
            ), "Wrong NMI behaviour, uncorrelated case."

    def test_smoke(self, device, dtype):
        """Basic functionality test"""
        img1 = torch.rand(100, device=device, dtype=dtype)
        img2 = torch.rand(100, device=device, dtype=dtype)

        loss = mutual_information_loss(img1, img2, num_bins=64)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

        normalized_loss = normalized_mutual_information_loss(img1, img2, num_bins=64)
        assert isinstance(normalized_loss, torch.Tensor)
        assert normalized_loss.shape == torch.Size([])

    def test_exception(self, device, dtype):
        """Test error conditions"""
        # Test with mismatched shapes
        img1 = torch.rand(10, device=device, dtype=dtype)
        img2 = torch.rand(20, device=device, dtype=dtype)

        with pytest.raises(Exception):
            mutual_information_loss(img1, img2)

        with pytest.raises(Exception):
            normalized_mutual_information_loss(img1, img2)

    def test_gradcheck(self, device):
        """Gradient checking"""
        img1 = torch.rand(50, device=device, dtype=torch.float64, requires_grad=True)
        img2 = torch.rand(50, device=device, dtype=torch.float64)

        self.gradcheck(mutual_information_loss, (img1, img2))
        self.gradcheck(normalized_mutual_information_loss, (img1, img2))

    def test_differentiability(self, device, dtype):
        torch.manual_seed(0)
        img_1 = self.sampling_function(10000, device, dtype)
        img_2 = self.sampling_function(10000, device, dtype)
        param = torch.tensor(1 / 2.0, requires_grad=True)
        mi = mutual_information_loss(img_1 + param * img_2, img_2)
        mi.backward()
        # negative gradient, order of magnitude 1/2
        assert -1 < param.grad < -1 / 10, f"Differentiability issue for mi, {param.grad=}."
        param = torch.tensor(1 / 2.0, requires_grad=True)
        nmi = normalized_mutual_information_loss(img_1 + param * img_2, img_2)
        nmi.backward()
        # negative gradient, order of magnitude 1/20
        assert -1 / 10 < param.grad < -1 / 100, f"Differentiability issue for nmi, {param.grad=}."

    def test_value_ranges(self, device, dtype):
        torch.manual_seed(0)
        self.value_ranges_check(device, dtype)

    def test_trivial_signal_normalizes_to_zero(self, device, dtype):
        signal = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], device=device, dtype=dtype)

        self.assert_close(_normalize_signal(signal, num_bins=64), torch.zeros_like(signal))

    def test_normalizes_onto_bin_centres(self, device, dtype):
        signal = torch.tensor([2.0, 2.5, 3.0], device=device, dtype=dtype)
        expected = torch.tensor([0.0, 31.5, 63.0], device=device, dtype=dtype)

        self.assert_close(_normalize_signal(signal, num_bins=64), expected)

    def test_joint_histogram_counts_every_sample(self, device, dtype):
        # The kernel weights of a sample sum to one over the bin centres, so the histogram counts every sample,
        # including the ones at a signal's maximum, only when the maximum is normalised onto the last centre.
        pred = torch.rand(2, 3, 20, device=device, dtype=dtype)
        target = torch.rand(2, 3, 20, device=device, dtype=dtype)
        module = MILossFromRef(target)
        joint_histogram = module._compute_joint_histogram(pred, module.eps)

        self.assert_close(
            joint_histogram.sum((-2, -1)), torch.full((2, 3), 20.0, device=device, dtype=dtype), low_tolerance=True
        )

    def test_binary_mask_mutual_information_is_its_entropy(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the entropies need full precision to match the hard histogram")
        generator = torch.Generator().manual_seed(0)
        mask = (torch.rand(12, 20, generator=generator) > 0.5).to(device=device, dtype=torch.float64)
        p = mask.mean()
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        mask = mask.to(dtype)

        self.assert_close(-mutual_information_loss_2d(mask, mask), entropy.to(dtype))
        self.assert_close(-mutual_information_loss_2d(1 - mask, mask), entropy.to(dtype))
        self.assert_close(
            -normalized_mutual_information_loss_2d(mask, mask), torch.tensor(2.0, device=device, dtype=dtype)
        )

    def test_integer_levels_match_hard_histogram(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the entropies need full precision to match the hard histogram")
        levels = 4
        pred = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 3, 3]], device=device)
        target = torch.tensor([[0, 1, 1, 3], [3, 3, 1, 0], [0, 2, 3, 3]], device=device)
        joint = torch.bincount(pred.flatten() * levels + target.flatten(), minlength=levels**2)
        joint = joint.view(levels, levels).double() / pred.numel()
        independent = joint.sum(-1, keepdim=True) * joint.sum(-2, keepdim=True)
        expected = torch.xlogy(joint, joint / independent).sum()

        mutual_information = -mutual_information_loss_2d(pred.to(dtype), target.to(dtype), num_bins=levels)
        inverted = -mutual_information_loss_2d((levels - 1 - pred).to(dtype), target.to(dtype), num_bins=levels)

        self.assert_close(mutual_information, expected.to(dtype))
        self.assert_close(inverted, mutual_information)

    @pytest.mark.parametrize("kernel", [MIKernel.xu, MIKernel.rectangular, MIKernel.truncated_gaussian])
    @pytest.mark.parametrize("dims", [(5,), (3, 1), (2, 8), (2, 1, 8), (2, 1, 2, 8), (2, 1, 2, 1, 8)])
    def test_batch_consistency(self, device, dtype, kernel, dims):
        torch.manual_seed(0)  # Fix seed for reproducibility

        img1 = torch.rand(dims, device=device, dtype=dtype)
        img2 = torch.rand(dims, device=device, dtype=dtype)

        # flatten batch dims
        unique_batch_dim_1 = img1.reshape((-1,) + img1.shape[-1:])
        unique_batch_dim_2 = img2.reshape((-1,) + img2.shape[-1:])

        # Compute batch loss
        loss_batch = mutual_information_loss(img1, img2, num_bins=64, kernel_function=kernel)
        normalized_loss_batch = normalized_mutual_information_loss(img1, img2, num_bins=64, kernel_function=kernel)

        # Compute iterative loss for verification
        losses = []
        normalized_losses = []
        for i in range(unique_batch_dim_1.shape[0]):
            loss = mutual_information_loss(
                unique_batch_dim_1[i], unique_batch_dim_2[i], num_bins=64, kernel_function=kernel
            )
            normalized_loss = normalized_mutual_information_loss(
                unique_batch_dim_1[i], unique_batch_dim_2[i], num_bins=64, kernel_function=kernel
            )
            losses.append(loss)
            normalized_losses.append(normalized_loss)

        loss_iterative = torch.stack(losses)
        normalized_loss_iterative = torch.stack(normalized_losses)

        # Compare
        assert loss_batch.shape == dims[:-1], (
            f"The shape of the batched losses for mi is wrong: {loss_batch.shape} vs {dims[:-1]}."
        )
        assert normalized_loss_batch.shape == dims[:-1], (
            f"The shape of the batched losses for nmi is wrong: {normalized_loss_batch.shape} vs {dims[:-1]}."
        )

        self.assert_close(loss_batch.flatten(), loss_iterative)
        self.assert_close(normalized_loss_batch.flatten(), normalized_loss_iterative)

    def test_module(self, device, dtype):
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (pred, target)

        op = normalized_mutual_information_loss
        op_module = NMILossFromRef(target)

        self.assert_close(op(*args), op_module(pred))

        op = mutual_information_loss
        op_module = MILossFromRef(target)

        self.assert_close(op(*args), op_module(pred))

    def test_masking(self, device, dtype):
        """test masking works on a 2d signal."""
        pred = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        target = torch.rand(2, 3, 64, 64, device=device, dtype=dtype)
        target_mask = torch.zeros(pred.shape[-2:], dtype=torch.bool, device=device)
        pred_mask = target_mask.clone()
        target_mask[:32] = True
        pred_mask[:, :32] = True
        # we tweak the values of target and pred for the normalization to be the same with or without the mask
        target[..., 0, 0] = 0
        target[..., 0, 1] = 1
        pred[..., 0, 0] = 0
        pred[..., 0, 1] = 1
        restricted_pred = pred[..., :32, :32]
        restricted_target = target[..., :32, :32]

        masked_kwargs = {"input": pred, "target": target, "input_mask": pred_mask, "target_mask": target_mask}
        restricted_kwargs = {
            "input": restricted_pred,
            "target": restricted_target,
        }
        self.assert_close(mutual_information_loss_2d(**masked_kwargs), mutual_information_loss_2d(**restricted_kwargs))
        self.assert_close(
            normalized_mutual_information_loss_2d(**masked_kwargs),
            normalized_mutual_information_loss_2d(**restricted_kwargs),
        )

    def test_dynamo(self, device, dtype, torch_optimizer):
        pred = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)
        target = torch.rand(2, 3, 3, 2, device=device, dtype=dtype)

        args = (pred, target)

        op = mutual_information_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(*args), op_optimized(*args), low_tolerance=True)

        op = normalized_mutual_information_loss
        op_optimized = torch_optimizer(op)

        self.assert_close(op(*args), op_optimized(*args), low_tolerance=True)
