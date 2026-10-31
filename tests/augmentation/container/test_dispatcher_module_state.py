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
from torch import nn

import kornia.augmentation as K

from testing.base import BaseTester


@pytest.mark.parametrize("dispatcher_class", [K.ManyToOneAugmentationDispather, K.ManyToManyAugmentationDispather])
class TestDispatcherModuleState(BaseTester):
    @staticmethod
    def call(dispatcher, image):
        if isinstance(dispatcher, K.ManyToManyAugmentationDispather):
            return dispatcher(*[(image,)] * len(dispatcher.augmentations))
        return dispatcher(image)

    def test_to_propagates_to_children(self, device, dtype, dispatcher_class):
        layers = [nn.Conv2d(2, 2, 1), nn.Conv2d(2, 2, 1)]
        dispatcher = dispatcher_class(*[K.AugmentationSequential(layer, data_keys=["input"]) for layer in layers])
        assert dispatcher.to(device=device, dtype=dtype) is dispatcher
        for layer in layers:
            assert layer.weight.device == device
            assert layer.weight.dtype == dtype
        image = torch.rand(2, 2, 4, 5, device=device, dtype=dtype)
        for actual, layer in zip(self.call(dispatcher, image), layers):
            self.assert_close(actual, layer(image), rtol=0, atol=0)

    def test_eval_propagates_without_updating_statistics(self, device, dtype, dispatcher_class):
        layers = [nn.BatchNorm2d(2).to(device=device, dtype=dtype) for _ in range(2)]
        dispatcher = dispatcher_class(*[K.AugmentationSequential(layer, data_keys=["input"]) for layer in layers])
        dispatcher.eval()
        assert all(not module.training for sequence in dispatcher.augmentations for module in sequence.modules())
        image = torch.arange(80, device=device, dtype=dtype).reshape(2, 2, 4, 5) / 80
        before = [layer.running_mean.clone() for layer in layers]
        actual = self.call(dispatcher, image)
        for output, layer, mean in zip(actual, layers, before):
            self.assert_close(layer.running_mean, mean, rtol=0, atol=0)
            self.assert_close(output, layer(image), rtol=0, atol=0)
        dispatcher.train()
        assert all(module.training for sequence in dispatcher.augmentations for module in sequence.modules())

    def test_parameters_and_checkpoint_roundtrip(self, device, dtype, dispatcher_class):
        layers = [nn.Conv2d(2, 2, 1).to(device=device, dtype=dtype) for _ in range(2)]
        dispatcher = dispatcher_class(*[K.AugmentationSequential(layer, data_keys=["input"]) for layer in layers])
        expected_parameters = [parameter for layer in layers for parameter in layer.parameters()]
        assert {id(p) for p in dispatcher.parameters()} == {id(p) for p in expected_parameters}
        state = copy.deepcopy(dispatcher.state_dict())
        assert len(state) == 4
        image = torch.rand(2, 2, 4, 5, device=device, dtype=dtype)
        expected = self.call(dispatcher, image)
        optimizer = torch.optim.SGD(dispatcher.parameters(), lr=0.1)
        sum(value.sum() for value in expected).backward()
        before = layers[0].weight.detach().clone()
        optimizer.step()
        assert not torch.equal(layers[0].weight, before)
        restored_layers = [nn.Conv2d(2, 2, 1).to(device=device, dtype=dtype) for _ in range(2)]
        restored = dispatcher_class(
            *[K.AugmentationSequential(layer, data_keys=["input"]) for layer in restored_layers]
        )
        restored.load_state_dict(state, strict=True)
        for actual, reference in zip(self.call(restored, image), expected):
            self.assert_close(actual, reference, rtol=0, atol=0)
