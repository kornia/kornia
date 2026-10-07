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
import warnings

import pytest
import torch

from kornia.core._compat import torch_version_lt
from kornia.core.exceptions import BaseError
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
            assert grad_pred.isfinite().all() and grad_target.isfinite().all()
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
        with pytest.raises(BaseError, match="same shape"):
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
        # A strided mask with exactly one sample's shape remains valid.
        storage = torch.ones((*shape, 2), device=device, dtype=torch.bool)
        mask = storage[..., 0]
        mask.reshape(-1)[1::3] = False
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
        assert actual.dtype == dtype and actual.device == device
        assert actual.isfinite().all()
        full_mask = torch.ones(shape, device=device, dtype=torch.bool)
        no_mask = loss_fn(signal, target, **kwargs)
        self.assert_close(no_mask, loss_fn(signal, target, full_mask, full_mask, **kwargs))
        self.assert_close(no_mask, module(target, **kwargs)(signal))
        assert no_mask.dtype == dtype and no_mask.device == device


class TestRectangularKernel(BaseTester):
    def test_dtype_and_values(self, device, dtype):
        signal = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=dtype)
        actual = rectangular_kernel(signal)
        assert actual.dtype == dtype and actual.device == device
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
