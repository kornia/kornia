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

from kornia.models.tiny_vit import BasicLayer, ConvLayer, TinyViT

from testing.base import BaseTester


def _small_model(stochastic: bool = False) -> TinyViT:
    return TinyViT(
        img_size=32,
        embed_dims=(8, 16, 24, 32),
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 3, 4),
        window_sizes=(2, 2, 2, 1),
        num_classes=3,
        drop_rate=0.1 if stochastic else 0.0,
        drop_path_rate=0.2 if stochastic else 0.0,
    ).eval()


class TestTinyViTCheckpoint(BaseTester):
    @pytest.mark.parametrize("training", [False, True])
    @pytest.mark.parametrize("first_trainable_stage", [0, 1])
    def test_frozen_prefix_retains_parameter_gradients(self, device, dtype, first_trainable_stage, training):
        # Fine-tuning can freeze the patch embedder or the first complete stage. The first remaining trainable
        # block then receives an activation without requires_grad; its parameters must still receive gradients.
        reference = _small_model(stochastic=training).train(training).to(device=device, dtype=dtype)
        reference.patch_embed.requires_grad_(False)
        for layer in list(reference.layers)[:first_trainable_stage]:
            layer.requires_grad_(False)
        checkpointed = copy.deepcopy(reference)
        for layer in checkpointed.layers:
            layer.use_checkpoint = True
        image = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)

        torch.manual_seed(731)
        expected = reference(image)
        torch.manual_seed(731)
        actual = checkpointed(image)
        self.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()

        expected_parameters = dict(reference.named_parameters())
        for name, parameter in checkpointed.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
                continue
            expected_gradient = expected_parameters[name].grad
            assert expected_gradient is not None, name
            assert parameter.grad is not None, name
            self.assert_close(parameter.grad, expected_gradient)

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="Input and second derivatives use full precision")
    def test_autograd_grad_and_second_derivatives(self, device, dtype):
        reference = _small_model().to(device=device, dtype=dtype)
        checkpointed = copy.deepcopy(reference)
        for layer in checkpointed.layers:
            layer.use_checkpoint = True
        image = torch.rand(2, 3, 32, 32, device=device, dtype=dtype, requires_grad=True)
        expected = reference(image)
        actual = checkpointed(image)
        expected_grad = torch.autograd.grad(expected.square().sum(), image, create_graph=True)[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), image, create_graph=True)[0]
        self.assert_close(actual_grad, expected_grad)
        expected_second = torch.autograd.grad(expected_grad.square().sum(), image)[0]
        actual_second = torch.autograd.grad(actual_grad.square().sum(), image)[0]
        self.assert_close(actual_second, expected_second)

    @pytest.mark.parametrize("stage", ["convolution", "attention"])
    def test_gradcheck(self, device, stage):
        if stage == "convolution":
            layer = ConvLayer(dim=2, depth=1, use_checkpoint=True).eval().to(device=device, dtype=torch.float64)
            data = torch.rand(1, 2, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        else:
            layer = BasicLayer(4, (2, 2), 1, 1, 2, use_checkpoint=True).eval().to(device=device, dtype=torch.float64)
            data = torch.rand(2, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(layer, (data,))

    @pytest.mark.applies_to(dtypes=("float32", "float64"), reason="Compiled gradient comparison uses full precision")
    @pytest.mark.parametrize("stage", ["convolution", "attention"])
    def test_dynamo(self, device, dtype, torch_optimizer, stage):
        if stage == "convolution":
            layer = ConvLayer(dim=2, depth=1, use_checkpoint=True).eval().to(device=device, dtype=dtype)
            data = torch.rand(1, 2, 2, 2, device=device, dtype=dtype)
        else:
            layer = BasicLayer(2, (2, 2), 1, 1, 2, use_checkpoint=True).eval().to(device=device, dtype=dtype)
            data = torch.rand(1, 4, 2, device=device, dtype=dtype)
        compiled = torch_optimizer(copy.deepcopy(layer))
        expected = layer(data)
        actual = compiled(data)
        self.assert_close(actual, expected)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for actual_parameter, expected_parameter in zip(compiled.parameters(), layer.parameters()):
            assert actual_parameter.grad is not None
            self.assert_close(actual_parameter.grad, expected_parameter.grad)
