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

import copy

import pytest
import torch

from kornia.feature import FilterResponseNorm2d, HyNet, LAFDescriptor, SOSNet, get_laf_descriptors
from kornia.feature._normalization import _DescriptorSequential
from kornia.feature.laf import extract_patches_from_pyramid, laf_from_center_scale_ori

from testing.base import BaseTester


class TestDescriptorNormalization(BaseTester):
    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    @pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16])
    def test_dynamo_laf_descriptor(self, device, torch_optimizer, model_cls, model_dtype):
        model = model_cls().to(device)
        norm = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)][-1]
        norm.running_var.fill_(1e5)
        model.to(model_dtype)
        wrapped = LAFDescriptor(model)
        img = torch.rand(1, 1, 48, 48, device=device, dtype=torch.float16)
        lafs = laf_from_center_scale_ori(
            torch.tensor([[[20.0, 20.0]]], device=device, dtype=img.dtype),
            torch.full((1, 1, 1, 1), 8.0, device=device, dtype=img.dtype),
        )
        self.assert_close(torch_optimizer(wrapped)(img, lafs), wrapped(img, lafs))

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    def test_default_half_dtype(self, device, model_cls):
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            with torch.device(device):
                model = model_cls()
        finally:
            torch.set_default_dtype(default_dtype)
        norm = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)][-1]
        assert norm.running_var.dtype == torch.float32
        norm.running_var.fill_(1e5)
        out = model(torch.rand(2, 1, 32, 32, device=device, dtype=torch.float16))
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    @pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("scripted", [False, True])
    def test_laf_descriptor_half_statistics(self, device, model_cls, model_dtype, scripted):
        model = model_cls().to(device)
        norm = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)][-1]
        norm.running_var.copy_(torch.linspace(1e5, 5e6, norm.num_features, device=device))
        model = model.to(model_dtype)
        reference = copy.deepcopy(model).half().eval()
        if scripted:
            model = torch.jit.script(model)
        model.train()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        modes = [layer.training for layer in model.modules()]
        img = torch.rand(1, 1, 48, 48, device=device, dtype=torch.float16)
        lafs = laf_from_center_scale_ori(
            torch.tensor([[[20.0, 20.0], [30.0, 30.0]]], device=device, dtype=img.dtype),
            torch.full((1, 2, 1, 1), 8.0, device=device, dtype=img.dtype),
        )
        patches = extract_patches_from_pyramid(img, lafs, 32).reshape(2, 1, 32, 32)
        expected = reference(patches).reshape(1, 2, -1)
        actual = LAFDescriptor(model)(img, lafs)
        self.assert_close(actual, expected)
        assert actual.dtype == img.dtype
        # Averaging keeps the loss gradient representable for the tiny, untrained descriptors.
        actual.float().mean().backward()
        expected.float().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            if parameter.grad is not None:
                self.assert_close(parameter.grad, ref_parameter.grad.to(parameter.dtype))
        with pytest.warns(UserWarning, match="LAF contains no keypoints"):
            empty = get_laf_descriptors(img, lafs[:, :0], model)
        assert empty.shape == (1, 0, 128)
        assert empty.dtype == img.dtype
        assert modes == [layer.training for layer in model.modules()]
        for name, tensor in model.state_dict().items():
            self.assert_close(tensor, before[name], rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    @pytest.mark.parametrize("convert", ["half", "to"])
    def test_half_preserves_running_statistics(self, device, model_cls, convert):
        model = model_cls().to(device)
        norm = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)][-1]
        # The published SOSNet and HyNet checkpoints contain variances up to 1.2e5 and 5.3e6.
        norm.running_var.copy_(torch.linspace(1e5, 5e6, norm.num_features, device=device))
        norm.running_mean.fill_(100.0)
        reference = copy.deepcopy(norm)
        reference_model = copy.deepcopy(model)
        state_keys = set(model.state_dict())

        model = model.half() if convert == "half" else model.to(dtype=torch.float16)
        assert set(model.state_dict()) == state_keys
        self.assert_close(norm.running_var, reference.running_var, rtol=0.0, atol=0.0)
        self.assert_close(norm.running_mean, reference.running_mean, rtol=0.0, atol=0.0)
        patches = torch.rand(2, 1, 32, 32, device=device)
        self.assert_close(model(patches.half()), reference_model(patches.half().float()).half())
        model.half().float()
        self.assert_close(norm.running_var, reference.running_var, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("is_eps_learnable", [False, True])
    def test_frn_large_half_activations(self, device, dtype, is_eps_learnable):
        layer = FilterResponseNorm2d(2, is_eps_leanable=is_eps_learnable).to(device, dtype)
        reference_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        reference = copy.deepcopy(layer).to(reference_dtype)
        # These magnitudes occur inside pretrained HyNet for ordinary [0, 1] image patches.
        x = torch.tensor([256.0, -512.0, 1024.0, -2048.0], device=device, dtype=dtype)
        x = x.repeat(2).reshape(1, 2, 2, 2).requires_grad_()
        ref_x = x.detach().to(reference_dtype).requires_grad_()
        actual = layer(x)
        expected = reference(ref_x)
        self.assert_close(actual, expected.to(dtype))
        actual.flatten()[0].backward()
        expected.flatten()[0].backward()
        # Compare gradients at unit scale so the dtype's absolute tolerance cannot hide zero gradients.
        self.assert_close(x.grad * 2048, ref_x.grad.to(dtype) * 2048)
        self.assert_close(layer.weight.grad, reference.weight.grad.to(dtype))
        self.assert_close(layer.bias.grad, reference.bias.grad.to(dtype))
        if is_eps_learnable:
            self.assert_close(layer.eps.grad, reference.eps.grad.to(dtype))

    @pytest.mark.parametrize("is_scale,is_bias", [(False, False), (False, True), (True, False), (True, True)])
    def test_frn_mixed_dtype(self, device, dtype, is_scale, is_bias):
        layer = FilterResponseNorm2d(2, is_scale=is_scale, is_bias=is_bias).to(device, dtype)
        x = torch.tensor([256.0, -512.0, 1024.0, -2048.0], device=device, dtype=torch.float16)
        x = x.repeat(2).reshape(1, 2, 2, 2)
        expected_dtype = torch.promote_types(x.dtype, dtype)
        reference = copy.deepcopy(layer).to(torch.float64 if dtype == torch.float64 else torch.float32)
        expected = reference(x.to(reference.eps.dtype)).to(expected_dtype)
        actual = layer(x)
        assert actual.dtype == expected_dtype
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    def test_half_load_state_dict(self, device, model_cls):
        source = model_cls().to(device)
        norm = [layer for layer in source.modules() if isinstance(layer, torch.nn.BatchNorm2d)][-1]
        norm.running_var.fill_(1e5)
        target = model_cls().to(device).half()
        target.load_state_dict(source.state_dict(), strict=True)
        source.half()
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float16)
        self.assert_close(target(patches), source(patches), rtol=0.0, atol=0.0)
        restored = model_cls().to(device).half()
        restored.load_state_dict(target.state_dict(), strict=True)
        self.assert_close(restored(patches), target(patches), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("momentum", [0.1, None])
    def test_final_batchnorm_state_and_gradients(self, device, training, momentum):
        model = _DescriptorSequential(torch.nn.Identity(), torch.nn.BatchNorm2d(2, affine=False, momentum=momentum))
        model = model.to(device).train(training)
        model[-1].running_var.fill_(1e5)
        reference = copy.deepcopy(model)
        model.half()
        x = torch.tensor([256.0, -512.0, 1024.0, -2048.0], device=device, dtype=torch.float16)
        x = x.repeat(4).reshape(2, 2, 2, 2).requires_grad_()
        ref_x = x.detach().float().requires_grad_()
        for _ in range(2):
            actual = model(x)
            expected = reference(ref_x).half()
            self.assert_close(actual, expected, rtol=0.0, atol=0.0)
        actual.flatten()[0].backward()
        expected.flatten()[0].backward()
        self.assert_close(x.grad, ref_x.grad.half())
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            self.assert_close(getattr(model[-1], name), getattr(reference[-1], name), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    @pytest.mark.parametrize("selection", [slice(None), slice(None, -1), slice(1, None), slice(0, 0), slice(None, 3)])
    def test_sequential_slices(self, device, model_cls, selection):
        model = model_cls().to(device)
        block = model.layers if model_cls is SOSNet else model.layer7
        block[-1].running_var.fill_(1e5)
        sliced = block[selection]
        reference = copy.deepcopy(sliced)
        shape = (2, 1, 32, 32) if model_cls is SOSNet else (2, 128, 8, 8)
        x = torch.rand(*shape, device=device, dtype=torch.float16)
        sliced.half()
        for layer in sliced.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                assert torch.isfinite(layer.running_var).all()
        if model_cls is SOSNet and selection == slice(None, 3):
            # An intermediate BatchNorm keeps ordinary Sequential half-precision behavior.
            reference = torch.nn.Sequential(*reference).half()
            self.assert_close(sliced(x), reference(x), rtol=0.0, atol=0.0)
        else:
            self.assert_close(sliced(x), reference(x.float()).half())
        assert set(sliced.state_dict()) == set(reference.state_dict())

    @pytest.mark.parametrize("model_cls", [SOSNet, HyNet])
    def test_slicing_does_not_change_source_statistics(self, device, model_cls):
        model = model_cls().to(device).half()
        block = model.layers if model_cls is SOSNet else model.layer7
        before = dict(model.named_buffers())
        patches = torch.rand(2, 1, 32, 32, device=device, dtype=torch.float16)
        expected = model(patches)
        for stop in range(1, len(block) + 1):
            block[:stop]
        for name, tensor in model.named_buffers():
            assert tensor is before[name]
        self.assert_close(model(patches), expected, rtol=0.0, atol=0.0)
        # Keeping the original protected norm works even when its position changes.
        block.append(torch.nn.Identity())
        block.insert(0, torch.nn.Identity())
        model.half()
        self.assert_close(model(patches), expected, rtol=0.0, atol=0.0)
