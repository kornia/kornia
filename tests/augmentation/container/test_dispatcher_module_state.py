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

    def test_moved_children_sample_like_a_moved_sequence(self, device, dtype, dispatcher_class):
        # After a move the children draw on that device and in that dtype, as an AugmentationSequential moved the
        # same way does. A CPU run with a non-default dtype checks the dtype half; a CUDA run checks the device.
        def sequences():
            return [
                K.AugmentationSequential(K.RandomHorizontalFlip(p=0.5), K.RandomBrightness((0.8, 1.2), p=1.0))
                for _ in range(2)
            ]

        dispatcher = dispatcher_class(*sequences()).to(device=device, dtype=dtype)
        reference = [sequence.to(device=device, dtype=dtype) for sequence in sequences()]
        for sequence in dispatcher.augmentations:
            for augmentation in sequence.children():
                assert augmentation.device.type == device.type
                assert augmentation.dtype == dtype
        image = torch.rand(4, 3, 4, 5, device=device, dtype=dtype)
        torch.manual_seed(0)
        actual = self.call(dispatcher, image)
        torch.manual_seed(0)
        expected = [sequence(image) for sequence in reference]
        for output, target in zip(actual, expected):
            self.assert_close(output, target, rtol=0, atol=0)

    def test_learnable_range_is_a_parameter(self, device, dtype, dispatcher_class):
        # Ranges stay on CPU in the default dtype unless the augmentation's RNG is moved.
        ranges = [nn.Parameter(torch.tensor([0.8, 1.2])) for _ in range(2)]
        dispatcher = dispatcher_class(
            *[K.AugmentationSequential(K.RandomBrightness(value, p=1.0), data_keys=["input"]) for value in ranges]
        )
        assert {id(parameter) for parameter in dispatcher.parameters()} == {id(value) for value in ranges}

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("learnable", [False, True])
    def test_state_dict_saved_before_registration_loads_strictly(
        self, device, dtype, dispatcher_class, nested, learnable
    ):
        ranges = [nn.Parameter(torch.tensor([0.8, 1.2])) if learnable else (0.8, 1.2) for _ in range(2)]
        dispatcher = dispatcher_class(
            *[K.AugmentationSequential(K.RandomBrightness(value, p=1.0), data_keys=["input"]) for value in ranges]
        ).to(device=device, dtype=dtype)
        module = nn.Sequential(nn.Identity(), dispatcher) if nested else dispatcher
        expected = copy.deepcopy(module.state_dict())
        assert expected

        # Before registration the dispatcher contributed no child state, including when nested.
        legacy = {}
        module.load_state_dict(legacy, strict=True)
        assert legacy == {}
        for name, value in module.state_dict().items():
            self.assert_close(value, expected[name], rtol=0, atol=0)

        # A checkpoint with any child state uses the new layout: do not hide a missing key.
        partial = dict(list(expected.items())[:1])
        with pytest.raises(RuntimeError, match="Missing key"):
            module.load_state_dict(partial, strict=True)

    @pytest.mark.parametrize("nested", [False, True])
    def test_current_state_dict_missing_all_child_state_is_reported(self, device, dtype, dispatcher_class, nested):
        dispatcher = dispatcher_class(
            *[
                K.AugmentationSequential(K.RandomBrightness(nn.Parameter(torch.tensor([0.8, 1.2])), p=1.0))
                for _ in range(2)
            ]
        ).to(device=device, dtype=dtype)
        module = nn.Sequential(nn.Identity(), dispatcher) if nested else dispatcher
        state = copy.deepcopy(module.state_dict())
        missing_keys = set(state)
        assert missing_keys
        # Removing the whole child subtree must not turn a current-format checkpoint into a legacy one.
        # Keep the state_dict metadata so the checkpoint's format remains identifiable.
        state.clear()

        with pytest.raises(RuntimeError, match="Missing key"):
            module.load_state_dict(state, strict=True)
        result = module.load_state_dict(state, strict=False)
        assert set(result.missing_keys) == missing_keys
        assert result.unexpected_keys == []

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("lazy_kind", ["linear", "batch_norm"])
    def test_legacy_state_keeps_lazy_children_uninitialized(self, device, dtype, dispatcher_class, nested, lazy_kind):
        layer = nn.LazyLinear(5) if lazy_kind == "linear" else nn.LazyBatchNorm2d()
        dispatcher = dispatcher_class(K.AugmentationSequential(layer, data_keys=["input"])).to(
            device=device, dtype=dtype
        )
        module = nn.Sequential(nn.Identity(), dispatcher) if nested else dispatcher
        uninitialized = {
            name: value for name, value in module.state_dict().items() if torch.nn.parameter.is_lazy(value)
        }
        assert uninitialized

        result = module.load_state_dict({}, strict=True)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        for name, value in module.state_dict().items():
            if name in uninitialized:
                assert value is uninitialized[name]
                assert torch.nn.parameter.is_lazy(value)
                assert value.device == device
                assert value.dtype == dtype

        image = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        output = self.call(dispatcher, image)
        assert len(output) == 1
        assert output[0].shape == image.shape
        assert output[0].device == device
        assert output[0].dtype == dtype
        assert torch.isfinite(output[0]).all()
        assert not any(torch.nn.parameter.is_lazy(value) for value in module.state_dict().values())

    @pytest.mark.device_agnostic
    @pytest.mark.parametrize("nested", [False, True])
    def test_non_tensor_extra_state_loads_legacy_and_roundtrips(self, dispatcher_class, nested):
        class ExtraStateModule(nn.Identity):
            def __init__(self, value):
                super().__init__()
                self.value = {"label": value}

            def get_extra_state(self):
                return self.value

            def set_extra_state(self, state):
                self.value = state

        layers = [ExtraStateModule("first"), ExtraStateModule("second")]
        dispatcher = dispatcher_class(*[K.AugmentationSequential(layer, data_keys=["input"]) for layer in layers])
        module = nn.Sequential(nn.Identity(), dispatcher) if nested else dispatcher
        expected = copy.deepcopy(module.state_dict())
        assert len(expected) == 2

        # The unregistered layout saved version-1 metadata without any child state.
        legacy = copy.deepcopy(expected)
        legacy.clear()
        prefixes = ("", "0", "1") if nested else ("",)
        legacy._metadata = {name: {"version": 1} for name in prefixes}
        module.load_state_dict(legacy, strict=True)
        assert legacy == {}
        assert module.state_dict() == expected

        for layer in layers:
            layer.value = {"label": "changed"}
        module.load_state_dict(expected, strict=True)
        assert module.state_dict() == expected
