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
import math
import warnings
from functools import cache

import pytest
import torch

from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import BaseError, ShapeError
from kornia.losses.mutual_information import (
    MIKernel,
    MILossFromRef,
    MILossFromRef2D,
    MILossFromRef3D,
    NMILossFromRef,
    NMILossFromRef2D,
    NMILossFromRef3D,
    _normalize_signal,
    mutual_information_loss,
    mutual_information_loss_2d,
    mutual_information_loss_3d,
    normalized_mutual_information_loss,
    normalized_mutual_information_loss_2d,
    normalized_mutual_information_loss_3d,
    rectangular_kernel,
)

from testing.base import BaseTester, dynamo_is_available


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

    @pytest.mark.parametrize("kernel", [MIKernel.xu, MIKernel.truncated_gaussian])
    def test_constant_signal_has_zero_gradient(self, device, dtype, kernel):
        # A constant sample is normalised to zero, and the discarded branch of that `where` must not backpropagate
        # 0 / 0: the loss does not change under any perturbation that keeps the range below eps, so its gradient is 0.
        generator = torch.Generator().manual_seed(0)
        pred = torch.rand(2, 12, 20, generator=generator).to(device, dtype)
        target = (pred**2 + 0.1 * torch.rand(2, 12, 20, generator=generator).to(device, dtype)).detach()
        pred[1] = 0.3
        for loss_fn in (mutual_information_loss_2d, normalized_mutual_information_loss_2d):
            pred_, target_ = pred.clone().requires_grad_(), target.clone().requires_grad_()
            grad_pred, grad_target = torch.autograd.grad(
                loss_fn(pred_, target_, kernel_function=kernel).sum(), (pred_, target_)
            )
            assert grad_pred.isfinite().all()
            assert grad_target.isfinite().all()
            self.assert_close(grad_pred[1], torch.zeros_like(grad_pred[1]), rtol=0, atol=0)
            assert grad_pred[0].abs().sum() > 0
        scale = torch.ones((), device=device, dtype=dtype, requires_grad=True)
        mutual_information_loss_2d(pred * scale, target, kernel_function=kernel).sum().backward()
        assert scale.grad.isfinite()
        reference = torch.full((240,), 0.5, device=device, dtype=dtype, requires_grad=True)
        MILossFromRef(reference, kernel_function=kernel)(target[0].reshape(-1)).backward()
        self.assert_close(reference.grad, torch.zeros_like(reference), rtol=0, atol=0)

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
        # the float64 oracle is computed on the CPU: MPS has no float64
        mask = (torch.rand(12, 20, generator=generator) > 0.5).double()
        p = mask.mean()
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        mask, entropy = mask.to(device, dtype), entropy.to(device, dtype)

        self.assert_close(-mutual_information_loss_2d(mask, mask), entropy)
        self.assert_close(-mutual_information_loss_2d(1 - mask, mask), entropy)
        self.assert_close(
            -normalized_mutual_information_loss_2d(mask, mask), torch.tensor(2.0, device=device, dtype=dtype)
        )

    def test_integer_levels_match_hard_histogram(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip("the entropies need full precision to match the hard histogram")
        levels = 4
        # the float64 oracle is computed on the CPU: MPS has no float64
        pred = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 3, 3]])
        target = torch.tensor([[0, 1, 1, 3], [3, 3, 1, 0], [0, 2, 3, 3]])
        joint = torch.bincount(pred.flatten() * levels + target.flatten(), minlength=levels**2)
        joint = joint.view(levels, levels).double() / pred.numel()
        independent = joint.sum(-1, keepdim=True) * joint.sum(-2, keepdim=True)
        expected = torch.xlogy(joint, joint / independent).sum().to(device, dtype)
        pred, target = pred.to(device, dtype), target.to(device, dtype)

        mutual_information = -mutual_information_loss_2d(pred, target, num_bins=levels)
        inverted = -mutual_information_loss_2d(levels - 1 - pred, target, num_bins=levels)

        self.assert_close(mutual_information, expected)
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

    @pytest.mark.parametrize("module", [MILossFromRef, NMILossFromRef])
    def test_to_dtype_matches_module_built_in_that_dtype(self, device, module):
        """``.to(dtype)`` makes the module use that dtype's epsilon, like one built in that dtype (#5547)."""
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        generator = torch.Generator().manual_seed(0)
        target = torch.rand(2, 12, 20, generator=generator).to(device)
        pred = torch.rand(2, 12, 20, generator=generator).to(device)

        moved = module(target).to(torch.float64)
        built = module(target.double())

        assert moved.eps == torch.finfo(torch.float64).eps
        self.assert_close(moved(pred.double()), built(pred.double()))

    @pytest.mark.parametrize("module", [MILossFromRef, NMILossFromRef])
    def test_to_device_moves_bin_centers(self, device, module):
        """``.to(device)`` moves ``bin_centers`` with the buffers and the ``state_dict`` keys stay the same (#5547)."""
        target = torch.rand(2, 3, 3, 2)
        pred = torch.rand(2, 3, 3, 2, device=device)

        mod = module(target).to(device)

        assert list(mod.state_dict().keys()) == ["signal", "mask"]
        assert mod.bin_centers.device == mod.signal.device
        self.assert_close(mod(pred), module(target.to(device))(pred))

        # The meta device stands in for an accelerator on a CPU-only runner.
        meta = module(target).to("meta")
        assert meta.bin_centers.device.type == meta.signal.device.type == "meta"

    @pytest.mark.parametrize("loss_fn", [mutual_information_loss, normalized_mutual_information_loss])
    @pytest.mark.parametrize("kernel", [MIKernel.xu, MIKernel.truncated_gaussian])
    def test_masked_gradcheck(self, device, loss_fn, kernel):
        generator = torch.Generator().manual_seed(5548)
        signal = torch.rand((2, 8), generator=generator, dtype=torch.float64).to(device).requires_grad_()
        target = torch.rand((2, 8), generator=generator, dtype=torch.float64).to(device).requires_grad_()
        mask = torch.tensor([True, False, True, True, False, True, True, True], device=device)
        self.gradcheck(lambda x, y: loss_fn(x, y, mask, mask, kernel, 4), (signal, target))


@pytest.mark.parametrize(
    "loss_fn,module,shape",
    [
        (mutual_information_loss, MILossFromRef, (12,)),
        (normalized_mutual_information_loss, NMILossFromRef, (12,)),
        (mutual_information_loss_2d, MILossFromRef2D, (3, 4)),
        (normalized_mutual_information_loss_2d, NMILossFromRef2D, (3, 4)),
        (mutual_information_loss_3d, MILossFromRef3D, (2, 3, 4)),
        (normalized_mutual_information_loss_3d, NMILossFromRef3D, (2, 3, 4)),
    ],
)
class TestMutualInformationValidation(BaseTester):
    @pytest.mark.parametrize("mask_dtype", [torch.int64, torch.int8, torch.float32])
    def test_mask_dtype(self, device, dtype, loss_fn, module, shape, mask_dtype):
        signal = torch.arange(1, 1 + torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)
        mask = (signal > 2).to(mask_dtype)
        with pytest.raises(BaseError, match=r"mask.*boolean"):
            loss_fn(signal, signal, input_mask=mask)
        with pytest.raises(BaseError, match=r"mask.*boolean"):
            loss_fn(signal, signal, target_mask=mask)
        with pytest.raises(BaseError, match=r"mask.*boolean"):
            module(signal, mask)
        with pytest.raises(BaseError, match=r"mask.*boolean"):
            module(signal)(signal, mask)

    def test_uint8_mask_reads_as_boolean(self, device, dtype, loss_fn, module, shape):
        # A uint8 0/1 mask selected the same samples as a boolean one before the dtype check (#5548): keep it,
        # without torch's deprecation warning for uint8 indices.
        signal = torch.linspace(0, 1, 2 * torch.Size(shape).numel(), device=device, dtype=dtype).reshape(2, *shape)
        target = signal.flip(-1) ** 2
        mask = torch.arange(torch.Size(shape).numel(), device=device).reshape(shape) % 3 != 1
        expected = loss_fn(signal, target, mask, mask)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = loss_fn(signal, target, mask.to(torch.uint8), mask.to(torch.uint8))
            from_ref = module(target, mask.to(torch.uint8))(signal, mask.to(torch.uint8))
        assert torch.equal(actual, expected)
        assert torch.equal(from_ref, expected)

    @pytest.mark.parametrize("kind", ["transposed", "batched", "singleton", "scalar"])
    def test_mask_shape(self, device, dtype, loss_fn, module, shape, kind):
        signal = torch.ones((2, *shape), device=device, dtype=dtype)
        mask = torch.ones(shape, device=device, dtype=torch.bool)
        if kind == "transposed":
            mask = mask.unsqueeze(0) if len(shape) == 1 else mask.transpose(-1, -2).contiguous()
        elif kind == "batched":
            mask = mask.expand(2, *shape)
        elif kind == "singleton":
            mask = mask.new_ones((1,) * len(shape))
        else:
            mask = mask.new_ones(())
        with pytest.raises(BaseError, match=r"mask.*one-sample shape"):
            loss_fn(signal, signal, input_mask=mask)
        with pytest.raises(BaseError, match=r"mask.*one-sample shape"):
            loss_fn(signal, signal, target_mask=mask)
        with pytest.raises(BaseError, match=r"mask.*one-sample shape"):
            module(signal, mask)
        with pytest.raises(BaseError, match=r"mask.*one-sample shape"):
            module(signal)(signal, mask)

    def test_signal_shape_before_flattening(self, device, dtype, loss_fn, module, shape):
        signal = torch.ones((2, *shape), device=device, dtype=dtype)
        other = signal.transpose(-1, -2).contiguous()
        with pytest.raises(ShapeError, match="same shape"):
            loss_fn(signal, other)
        # The generic module retains its documented ValueError for incompatible signal shapes.
        with pytest.raises((BaseError, ValueError), match="shape"):
            module(signal)(other)

    @pytest.mark.parametrize("kind", ["input_empty", "target_empty", "disjoint"])
    def test_empty_mask_intersection(self, device, dtype, loss_fn, module, shape, kind):
        signal = torch.arange(torch.Size(shape).numel(), device=device, dtype=dtype).reshape(shape)
        input_mask = signal < 2
        target_mask = ~input_mask
        if kind == "input_empty":
            input_mask = torch.zeros_like(input_mask)
            target_mask = None
        elif kind == "target_empty":
            target_mask = torch.zeros_like(target_mask)
            input_mask = None
        with pytest.raises(BaseError, match=r"mask.*at least one sample"):
            loss_fn(signal, signal, input_mask, target_mask)
        with pytest.raises(BaseError, match=r"mask.*at least one sample"):
            module(signal, target_mask)(signal, input_mask)

    @pytest.mark.parametrize("num_bins", [0, 1, -1, 2.5, 2.0, True])
    def test_num_bins(self, device, dtype, loss_fn, module, shape, num_bins):
        signal = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="num_bins must be an integer >= 2"):
            loss_fn(signal, signal, num_bins=num_bins)
        with pytest.raises(BaseError, match="num_bins must be an integer >= 2"):
            module(signal, num_bins=num_bins)

    @pytest.mark.parametrize("window_radius", [0.0, -1.0])
    def test_window_radius(self, device, dtype, loss_fn, module, shape, window_radius):
        signal = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(BaseError, match="window_radius must be > 0"):
            loss_fn(signal, signal, window_radius=window_radius)
        with pytest.raises(BaseError, match="window_radius must be > 0"):
            module(signal, window_radius=window_radius)

    @pytest.mark.parametrize("kernel", ["xu", MIKernel.xu.value, None, 0])
    def test_kernel_member(self, device, dtype, loss_fn, module, shape, kernel):
        signal = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="kernel_function must be a MIKernel member"):
            loss_fn(signal, signal, kernel_function=kernel)
        with pytest.raises(ValueError, match="kernel_function must be a MIKernel member"):
            module(signal, kernel_function=kernel)

    @pytest.mark.parametrize("kernel", list(MIKernel))
    def test_valid_masks_and_boundaries(self, device, dtype, loss_fn, module, shape, kernel):
        generator = torch.Generator().manual_seed(5548)
        signal = torch.rand((2, *shape), generator=generator).to(device, dtype)
        target = torch.rand((2, *shape), generator=generator).to(device, dtype)
        # A non-contiguous mask with exactly one sample's shape remains valid; an image mask is laid out so that it
        # cannot be viewed flat.
        mask = torch.ones(shape, device=device, dtype=torch.bool)
        mask.view(-1)[1::3] = False
        mask = torch.stack([mask, mask], -1)[..., 0] if len(shape) == 1 else mask.mT.contiguous().mT
        assert not mask.is_contiguous()
        kwargs = {"kernel_function": kernel, "num_bins": 2, "window_radius": 0.5}
        actual = loss_fn(signal, target, mask, mask, **kwargs)
        flat_loss = (
            normalized_mutual_information_loss
            if module in (NMILossFromRef, NMILossFromRef2D, NMILossFromRef3D)
            else mutual_information_loss
        )
        expected = flat_loss(
            signal.reshape(2, -1)[:, mask.reshape(-1)], target.reshape(2, -1)[:, mask.reshape(-1)], **kwargs
        )
        self.assert_close(actual, expected)
        self.assert_close(actual, module(target, mask, **kwargs)(signal, mask))
        assert actual.dtype == dtype
        assert actual.device == device
        assert actual.isfinite().all()
        full_mask = torch.ones(shape, device=device, dtype=torch.bool)
        no_mask = loss_fn(signal, target, **kwargs)
        self.assert_close(no_mask, loss_fn(signal, target, full_mask, full_mask, **kwargs))
        self.assert_close(no_mask, module(target, **kwargs)(signal))
        assert no_mask.dtype == dtype
        assert no_mask.device == device


class TestRectangularKernel(BaseTester):
    def test_dtype_and_values(self, device, dtype):
        signal = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=dtype)
        actual = rectangular_kernel(signal)
        assert actual.dtype == dtype
        assert actual.device == device
        self.assert_close(actual, signal.new_tensor([0, 1, 1, 1, 0]))


@pytest.mark.skipif(not dynamo_is_available() or torch_version_lt(2, 9, 0), reason="Dynamo traces the gathers from 2.9")
@pytest.mark.parametrize(
    "loss_fn,shape",
    [
        (mutual_information_loss, (12,)),
        (normalized_mutual_information_loss, (12,)),
        (mutual_information_loss_2d, (3, 4)),
        (normalized_mutual_information_loss_2d, (3, 4)),
        (mutual_information_loss_3d, (2, 3, 4)),
        (normalized_mutual_information_loss_3d, (2, 3, 4)),
    ],
)
class TestMutualInformationEagerBackendTraces(BaseTester):
    # The names keep "compile" and "dynamo" out, so the ordinary jobs run these.
    def test_eager_backend_traces_without_a_mask(self, device, dtype, loss_fn, shape):
        # Without a mask the losses trace in one graph: the empty-mask check must not guard on the size of the
        # unmasked gather.
        signal = torch.linspace(0, 1, 2 * torch.Size(shape).numel(), device=device, dtype=dtype).reshape(2, *shape)
        target = signal.flip(-1) ** 2
        torch._dynamo.reset()
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            compiled = torch.compile(loss_fn, backend="eager", fullgraph=True)
            self.assert_close(compiled(signal, target), loss_fn(signal, target))


_HALF_FLOOR_SKIP = (
    "MI in half precision is biased by the empty-bin floor finfo(dtype).eps, applied in count units before the "
    "histogram is normalised (#4153)"
)


@cache
def _supports_bool_index_backward(device_type: str, dtype: torch.dtype) -> bool:
    """Whether boolean-mask indexing has a backward kernel here (torch 2.5.1 has none for MPS half).

    The losses index the target with its mask, so a gradient to the target needs it.
    """
    x = torch.zeros(2, device=device_type, dtype=dtype, requires_grad=True)
    try:
        x[torch.ones(2, dtype=torch.bool, device=device_type)].sum().backward()
    except RuntimeError:
        return False
    return True


class TestConventionsMutualInformation(BaseTester):
    @staticmethod
    def _images(device, dtype):
        """A batch of two asymmetric 12 x 20 images: a column ramp plus noise, a nonlinear function of it, noise."""
        g = torch.Generator().manual_seed(0)
        a = torch.rand(2, 12, 20, generator=g, dtype=torch.float64) + torch.linspace(0, 1, 20, dtype=torch.float64)
        c = a**2 + 0.05 * torch.rand(2, 12, 20, generator=g, dtype=torch.float64)
        b = torch.rand(2, 12, 20, generator=g, dtype=torch.float64)
        return tuple(t.to(device=device, dtype=dtype) for t in (a, c, b))

    @staticmethod
    def _masks(device):
        """Input ROI = left 14 columns, target ROI = top 8 rows of a 12 x 20 image (112 pixels in both)."""
        input_mask = torch.zeros(12, 20, dtype=torch.bool)
        input_mask[:, :14] = True
        target_mask = torch.zeros(12, 20, dtype=torch.bool)
        target_mask[:8] = True
        return input_mask.to(device), target_mask.to(device)

    @staticmethod
    def _levels(device):
        """Integer images with levels 0 .. 8, both extremes present in each."""
        i = torch.arange(12, device=device)[:, None]
        j = torch.arange(20, device=device)[None, :]
        a = (i + 2 * j) % 9
        return a, (2 * a + i % 3) % 9

    def test_convention_mi_losses_are_negative_mi_and_nmi_in_nats(self, device, dtype):
        a, c, b = self._images(device, dtype)
        x, y = a.flatten(-2), c.flatten(-2)
        # the module built on the target returns the marginal entropies of the joint histogram and its joint entropy
        h_1, h_2, h_12 = MILossFromRef(y).entropies(x)
        self.assert_close(mutual_information_loss(x, y), -(h_1 + h_2 - h_12))
        self.assert_close(normalized_mutual_information_loss(x, y), -(h_1 + h_2) / h_12)
        # minimising the loss maximises the dependence: identical < dependent < independent, per sample
        mi = [mutual_information_loss_2d(a, t) for t in (a, c, b)]
        nmi = [normalized_mutual_information_loss_2d(a, t) for t in (a, c, b)]
        for losses in (mi, nmi):
            assert (losses[0] < losses[1]).all()
            assert (losses[1] < losses[2]).all()
        # NMI = (H_1 + H_2) / H_12 lies in [1, 2], so its loss lies in [-2, -1]
        nmi = torch.stack(nmi)
        assert ((nmi >= -2) & (nmi <= -1)).all()
        # natural log: a ramp fills the bins evenly, so each marginal entropy is close to ln(num_bins) nats
        # (in bits it would be 4 and 6)
        ramp = torch.linspace(0, 1, 4096, device=device, dtype=dtype)
        for num_bins in (16, 64):
            h_1, h_2, _ = MILossFromRef(ramp, num_bins=num_bins).entropies(ramp)
            assert abs(h_1.item() - math.log(num_bins)) < 0.05
            assert abs(h_2.item() - math.log(num_bins)) < 0.05
        # entropies() returns the compared signal's marginal first: a cubed ramp crowds the low bins
        h_1, h_2, _ = MILossFromRef(ramp).entropies(ramp**3)
        assert h_1 < h_2 - 0.1

    def test_convention_mi_losses_are_symmetric_and_ignore_pixel_order(self, device, dtype):
        a, c, _ = self._images(device, dtype)
        x, y = a.flatten(-2), c.flatten(-2)
        perm = torch.randperm(240, generator=torch.Generator().manual_seed(1)).to(device)
        for loss in (mutual_information_loss, normalized_mutual_information_loss):
            expected = loss(x, y)
            self.assert_close(loss(y, x), expected)
            # relabelling the pixels of both images alike changes nothing ...
            self.assert_close(loss(x[..., perm], y[..., perm]), expected)
            # ... while permuting one image only destroys the correspondence (control)
            assert (loss(x[..., perm], y) - expected > 0.1).all()
        # transposing both images (H != W) leaves the 2-D losses unchanged
        self.assert_close(mutual_information_loss_2d(a.mT, c.mT), mutual_information_loss_2d(a, c))
        self.assert_close(
            normalized_mutual_information_loss_2d(a.mT, c.mT), normalized_mutual_information_loss_2d(a, c)
        )

    def test_convention_mi_losses_min_max_normalise_each_sample(self, device, dtype):
        a, c, _ = self._images(device, dtype)
        expected = mutual_information_loss_2d(a, c)
        # positive affine maps of sample 1 in both images: no assumed data range (sample 1 unchanged) and no
        # batch-wide range (sample 0 bitwise unchanged)
        x, y = a.clone(), c.clone()
        x[1] = x[1] * 1000 + 50
        y[1] = y[1] * 1000 - 50
        loss = mutual_information_loss_2d(x, y)
        assert torch.equal(loss[0], expected[0])
        # half precision rounds the mapped values to a coarser grid of bin positions
        self.assert_close(loss[1], expected[1], low_tolerance=dtype in (torch.float16, torch.bfloat16))
        # one outlier stretches its sample's range and squeezes every other value into a few bins
        x = a.clone()
        x[0, 3, 5] = 50.0
        loss = mutual_information_loss_2d(x, c)
        assert loss[0] - expected[0] > 1.0
        assert torch.equal(loss[1], expected[1])

    def test_convention_mi_losses_treat_a_range_below_eps_as_constant(self, device, dtype):
        a, c, _ = self._images(device, dtype)
        eps = torch.finfo(dtype).eps
        constant = torch.full_like(c, 0.3)
        # the threshold is absolute: a * eps / 4 spans about eps / 2, although its relative range is about 1
        tiny = a * (eps / 4)
        for x, y in ((a, constant), (constant, c), (tiny, c)):
            # num_bins=16 keeps the empty-bin floor small in half precision; the rule does not depend on it
            mi = mutual_information_loss_2d(x, y, num_bins=16)
            nmi = normalized_mutual_information_loss_2d(x, y, num_bins=16)
            assert (mi.abs() < 0.05).all()
            assert ((nmi + 1).abs() < 0.05).all()
        # control: at 64 eps per unit the same image is a signal (a power-of-two scaling changes no position)
        self.assert_close(
            mutual_information_loss_2d(a * (64 * eps), c, num_bins=16), mutual_information_loss_2d(a, c, num_bins=16)
        )
        # the threshold includes eps itself: levels 0, eps / 2 and eps are constant, levels 0, eps and 2 eps are not
        med = c.flatten(-2).median(-1).values[:, None, None]
        for k, is_constant in ((1, True), (2, False)):
            t = torch.where(c < med, 0.0, k * eps / 2).to(dtype)
            t[:, 0, 0] = k * eps
            mi = mutual_information_loss_2d(t, c, num_bins=16)
            assert (mi.abs() < 0.05).all() if is_constant else (mi < -0.3).all()

    def test_convention_mi_losses_return_one_value_per_leading_index(self, device, dtype):
        g = torch.Generator().manual_seed(2)
        x = torch.rand(2, 3, 6, 10, generator=g, dtype=torch.float64)
        y = x**2 + 0.1 * torch.rand(2, 3, 6, 10, generator=g, dtype=torch.float64)
        x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)
        pairs = (
            (mutual_information_loss_2d, mutual_information_loss_3d),
            (normalized_mutual_information_loss_2d, normalized_mutual_information_loss_3d),
        )
        for loss_2d, loss_3d in pairs:
            # no reduction: _2d reads (B, C, H, W) as B x C images, _3d as B volumes of depth C
            out_2d, out_3d = loss_2d(x, y), loss_3d(x, y)
            assert out_2d.shape == (2, 3)
            assert out_3d.shape == (2,)
            assert out_2d.dtype == dtype
            assert out_3d.dtype == dtype
            for i in range(2):
                self.assert_close(out_3d[i], loss_3d(x[i], y[i]))
                for j in range(3):
                    self.assert_close(out_2d[i, j], loss_2d(x[i, j], y[i, j]))
        # one target is not broadcast over a batch of inputs
        with pytest.raises((ValueError, BaseError)):
            mutual_information_loss(x[:, 0].flatten(-2), y[0, 0].flatten())

    @pytest.mark.parametrize(
        "flat, loss_2d, loss_3d",
        [
            (mutual_information_loss, mutual_information_loss_2d, mutual_information_loss_3d),
            (
                normalized_mutual_information_loss,
                normalized_mutual_information_loss_2d,
                normalized_mutual_information_loss_3d,
            ),
        ],
    )
    def test_convention_mi_losses_2d_3d_flatten_the_trailing_axes(self, device, dtype, flat, loss_2d, loss_3d):
        a, c, _ = self._images(device, dtype)
        input_mask, target_mask = self._masks(device)
        # _2d is the flat loss on the row-major flattened last two axes; its masks are one image's (H, W)
        assert torch.equal(
            loss_2d(a, c, input_mask, target_mask),
            flat(a.flatten(-2), c.flatten(-2), input_mask.flatten(), target_mask.flatten()),
        )
        # _3d flattens the last three axes; volumes of D = 4, H = 3, W = 20 with (D, H, W) masks
        v, w = a.reshape(2, 4, 3, 20), c.reshape(2, 4, 3, 20)
        v_mask, w_mask = input_mask.reshape(4, 3, 20), target_mask.reshape(4, 3, 20)
        assert torch.equal(
            loss_3d(v, w, v_mask, w_mask), flat(v.flatten(-3), w.flatten(-3), v_mask.flatten(), w_mask.flatten())
        )
        # a volume of depth 1 is an image
        assert torch.equal(loss_3d(a[:, None], c[:, None]), loss_2d(a, c))

    @pytest.mark.parametrize(
        "loss, module, shape",
        [
            (mutual_information_loss, MILossFromRef, (240,)),
            (mutual_information_loss_2d, MILossFromRef2D, (12, 20)),
            (mutual_information_loss_3d, MILossFromRef3D, (4, 3, 20)),
            (normalized_mutual_information_loss, NMILossFromRef, (240,)),
            (normalized_mutual_information_loss_2d, NMILossFromRef2D, (12, 20)),
            (normalized_mutual_information_loss_3d, NMILossFromRef3D, (4, 3, 20)),
        ],
    )
    def test_convention_mi_losses_equal_the_from_ref_module_built_on_the_target(
        self, device, dtype, loss, module, shape
    ):
        a, c, _ = self._images(device, dtype)
        input_mask, target_mask = self._masks(device)
        x, y = a.reshape(2, *shape), c.reshape(2, *shape)
        x_mask, y_mask = input_mask.reshape(shape), target_mask.reshape(shape)
        # bitwise: the target is the cached reference, the input the forward argument
        assert torch.equal(loss(x, y, x_mask, y_mask), module(y, y_mask)(x, x_mask))

    def test_convention_mi_losses_histogram_the_mask_intersection_normalised_per_own_mask(self, device, dtype):
        a, c, _ = self._images(device, dtype)
        input_mask, target_mask = self._masks(device)
        expected = mutual_information_loss_2d(a, c, input_mask, target_mask)

        def loss(x, y):
            return mutual_information_loss_2d(x, y, input_mask, target_mask)

        # pixel (11, 19) is outside both masks: extreme values there change nothing
        x, y = a.clone(), c.clone()
        x[:, 11, 19], y[:, 11, 19] = 1000.0, -1000.0
        assert torch.equal(loss(x, y), expected)
        # pixel (10, 0) is inside the input mask only: an in-range value there is not in the joint histogram ...
        both = input_mask & target_mask
        x = a.clone()
        x[:, 10, 0] = (a[:, both].amin(-1) + a[:, both].amax(-1)) / 2
        assert torch.equal(loss(x, c), expected)
        # ... but the pixel takes part in the input's min-max normalisation, so a new maximum there moves the loss
        x[:, 10, 0] = 5.0
        assert ((loss(x, c) - expected).abs() > 0.1).all()
        # likewise pixel (0, 19), inside the target mask only, as a new minimum of the target
        y = c.clone()
        y[:, 0, 19] = -5.0
        assert ((loss(a, y) - expected).abs() > 0.1).all()
        # the input is normalised over its own mask only: an extreme input value at the target-only pixel (0, 19)
        # changes nothing, and an in-range target value there is not in the joint histogram
        x = a.clone()
        x[:, 0, 19] = 1000.0
        assert torch.equal(loss(x, c), expected)
        y = c.clone()
        y[:, 0, 19] = (c[:, both].amin(-1) + c[:, both].amax(-1)) / 2
        assert torch.equal(loss(a, y), expected)
        # masks have one image's (H, W) layout: transposing the images together with their masks changes nothing
        transposed = mutual_information_loss_2d(a.mT, c.mT, input_mask.T.contiguous(), target_mask.T.contiguous())
        self.assert_close(transposed, expected)

    def test_convention_mi_from_ref_caches_a_normalised_copy_of_the_reference(self, device, dtype):
        a, c, b = self._images(device, dtype)
        input_mask, target_mask = self._masks(device)
        reference = c.clone()
        module = MILossFromRef2D(reference, target_mask)
        # bin_centers is a non-persistent buffer, so the state_dict holds the reference and its mask only
        assert sorted(module.state_dict()) == ["mask", "signal"]
        expected = module(a, input_mask)
        # editing the reference after construction is not seen
        reference.mul_(3).add_(b)
        assert torch.equal(module(a, input_mask), expected)
        assert ((MILossFromRef2D(reference, target_mask)(a, input_mask) - expected).abs() > 0.1).all()
        # the cache is not detached: a reference that requires grad gets one, and only one, backward pass
        if not _supports_bool_index_backward(device.type, dtype):
            pytest.skip("no boolean-index backward kernel for this device and dtype")
        reference = c.clone().requires_grad_(True)
        module = MILossFromRef2D(reference)
        module(a).sum().backward()
        assert reference.grad is not None
        assert reference.grad.abs().sum() > 0
        with pytest.raises(RuntimeError):
            module(a).sum().backward()

    def test_convention_mi_kernel_members_hold_the_kernel_functions(self, device, dtype):
        assert [k.name for k in MIKernel] == ["xu", "rectangular", "truncated_gaussian"]
        d = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], device=device, dtype=dtype)
        # the members are not callable; their value is the kernel f(d, window_radius=1.0)
        with pytest.raises(TypeError):
            MIKernel.xu(d)
        # xu (Xu et al. 2008, Eq. 22): 1 - 0.1|d| - 1.8 d^2 below |d| = 0.5, 1.9 - 3.7|d| + 1.8 d^2 up to |d| = 1;
        # rectangular: 1 on the closed support; truncated_gaussian: N(0, 1) density cut at |d| = 1
        gaussian = [math.exp(-(v**2) / 2) / math.sqrt(2 * math.pi) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
        values = {
            "xu": [1.0, 0.8625, 0.5, 0.1375, 0.0, 0.0],
            "rectangular": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "truncated_gaussian": [*gaussian, 0.0],
        }
        tol = 4 * torch.finfo(dtype).eps
        for kernel in MIKernel:
            actual = kernel.value(d)
            expected = torch.tensor(values[kernel.name], device=device, dtype=actual.dtype)
            self.assert_close(actual, expected, rtol=0.0, atol=tol)
            # window_radius stretches the support; it is also the Gaussian's sigma, which halves that density
            scale = 0.5 if kernel is MIKernel.truncated_gaussian else 1.0
            self.assert_close(kernel.value(2 * d, window_radius=2.0), expected * scale, rtol=0.0, atol=tol)
        # at radius 1, xu weights over unit-spaced bin centres sum to 1 (0.8625 + 0.1375, 0.5 + 0.5)
        centres = torch.arange(64, device=device).to(dtype)
        t = torch.tensor([0.25, 17.75, 40.5, 62.75], device=device, dtype=dtype)
        self.assert_close(MIKernel.xu.value(centres[:, None] - t).sum(0), torch.ones_like(t), rtol=0.0, atol=tol)

    @pytest.mark.parametrize("kernel", list(MIKernel))
    def test_convention_mi_losses_differentiate_both_arguments_except_rectangular(self, device, dtype, kernel):
        if kernel is not MIKernel.rectangular and not _supports_bool_index_backward(device.type, dtype):
            pytest.skip("no boolean-index backward kernel for this device and dtype")
        a, c, _ = self._images(device, dtype)
        x, y = a.requires_grad_(True), c.requires_grad_(True)
        for loss_fn in (mutual_information_loss_2d, normalized_mutual_information_loss_2d):
            loss = loss_fn(x, y, kernel_function=kernel)
            if kernel is MIKernel.rectangular:
                # the box kernel is piecewise constant: the loss has no autograd graph (evaluation only)
                assert not loss.requires_grad
                with pytest.raises(RuntimeError):
                    loss.sum().backward()
            else:
                # the target is rebuilt on every call, so both arguments get a gradient, in every sample
                grad_x, grad_y = torch.autograd.grad(loss.sum(), (x, y))
                for grad in (grad_x, grad_y):
                    assert torch.isfinite(grad).all()
                    assert (grad.abs().flatten(1).sum(1) > 0).all()

    def test_convention_mi_losses_num_bins_and_window_radius_set_the_resolution(self, device, dtype):
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_FLOOR_SKIP)
        a, c, _ = self._images(device, dtype)
        # finer bins raise MI, a wider kernel lowers it: values compare only at equal settings
        mi_aa = [mutual_information_loss_2d(a, a, num_bins=n) for n in (4, 16, 64)]
        assert (mi_aa[0] > mi_aa[1]).all()
        assert (mi_aa[1] > mi_aa[2]).all()
        mi_ac = [mutual_information_loss_2d(a, c, window_radius=r) for r in (0.5, 1.0, 2.0, 4.0)]
        assert all((low < high).all() for low, high in zip(mi_ac, mi_ac[1:]))
        # the default histogram is soft: an image is not fully informative about itself, MI(a, a) < H(a) and
        # NMI(a, a) < 2 ...
        h_a, _, _ = MILossFromRef(a.flatten(-2)).entropies(a.flatten(-2))
        assert (-mutual_information_loss_2d(a, a) < h_a - 0.1).all()
        assert (normalized_mutual_information_loss_2d(a, a) > -1.9).all()
        # ... while window_radius=0.5 gives every sample a single bin, a hard histogram: NMI(a, a) = 2
        self.assert_close(
            normalized_mutual_information_loss_2d(a, a, window_radius=0.5),
            torch.full((2,), -2.0, device=device, dtype=dtype),
        )

    def test_convention_mi_losses_match_hard_histograms_on_integer_levels_5546(self, device, dtype):
        """Integer levels 0 .. K on the bin centres (``num_bins=K + 1``) give the hard-histogram MI and NMI (#5546)."""
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(_HALF_FLOOR_SKIP + "; float16 entropies also miss the reference literals' tolerance")
        # levels 0 .. 8 with both extremes present normalise onto the centres 0 .. 8 of num_bins=9, where the
        # default kernel puts every pixel, the maximum included, in its own bin. scikit-learn 1.9.0 /
        # scikit-image 0.26.0 (numpy 2.0.0) on the same arrays:
        #   mutual_info_score(A.ravel(), B.ravel()) = 1.0991837969703617
        #   normalized_mutual_information(A, B, bins=9) = 1.3338539514099725
        levels_a, levels_b = self._levels(device)
        x, y = levels_a.to(dtype), levels_b.to(dtype)
        mi = mutual_information_loss_2d(x, y, num_bins=9)
        self.assert_close(mi, torch.tensor(-1.0991837969703617, device=device, dtype=dtype))
        self.assert_close(
            normalized_mutual_information_loss_2d(x, y, num_bins=9),
            torch.tensor(-1.3338539514099725, device=device, dtype=dtype),
        )
        # every pixel counts, so MI does not depend on the intensity polarity
        self.assert_close(mutual_information_loss_2d(8 - x, y, num_bins=9), mi)

    def test_convention_mi_from_ref_follows_to_dtype_5547(self, device, dtype):
        """A ``FromRef`` module moved with ``.to(dtype)`` computes like one built in that dtype (#5547)."""
        # each move crosses a large gap in eps (MPS has no float64), so a module that kept its construction eps,
        # the empty-bin floor, would compute a different loss (into float16: NaN)
        if dtype in (torch.float16, torch.bfloat16):
            moved_dtype = torch.float32
        elif dtype == torch.float32 and device.type != "mps":
            moved_dtype = torch.float64
        else:
            moved_dtype = torch.float16
        # integer levels 0 .. 4 normalise to multiples of (num_bins - 1) / 4 exactly in every dtype, so the cached
        # reference converts exactly and any difference comes from the module's own state
        reference = self._levels(device)[0] % 5
        # eps acts here as the empty-bin floor (#4153); re-derive the fixture if the floor changes
        moved = MILossFromRef2D(reference.to(dtype)).to(moved_dtype)
        native = MILossFromRef2D(reference.to(moved_dtype))
        assert torch.equal(moved.signal, native.signal)
        x = self._images(device, moved_dtype)[0][0]
        assert torch.equal(moved(x), native(x))

    def test_wart_mi_losses_float16_nan_on_a_large_image_4153(self, device, dtype):
        """The empty-bin floor of ``finfo(dtype).eps`` counts makes float16 NaN on large images (#4153)."""
        if dtype != torch.float16:
            pytest.skip("float16 only: the floor eps / mass rounds to 0 once the mass passes 2**15")
        g = torch.Generator().manual_seed(7)
        u = torch.rand(184, 184, generator=g, dtype=torch.float64)
        v = u**2 + 0.05 * torch.rand(184, 184, generator=g, dtype=torch.float64)
        x, y = u.to(device, dtype), v.to(device, dtype)
        assert torch.isnan(mutual_information_loss_2d(x, y)).all()
        # control: a smaller image in float16 and the same image in float32 are finite
        assert torch.isfinite(mutual_information_loss_2d(x[:120, :120], y[:120, :120])).all()
        assert torch.isfinite(mutual_information_loss_2d(x.float(), y.float())).all()
        # the rectangular kernel counts each sample in about two bins per signal, four times xu's histogram mass
        rect = mutual_information_loss_2d(x[:100, :100], y[:100, :100], kernel_function=MIKernel.rectangular)
        assert torch.isnan(rect).all()
