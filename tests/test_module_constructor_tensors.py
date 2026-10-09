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

import kornia


def _module(case, value):
    constructors = {
        "dice": lambda: kornia.losses.DiceLoss(weight=value),
        "focal": lambda: kornia.losses.FocalLoss(alpha=0.25, weight=value),
        "binary_weight": lambda: kornia.losses.BinaryFocalLossWithLogits(alpha=0.25, weight=value),
        "binary_pos_weight": lambda: kornia.losses.BinaryFocalLossWithLogits(alpha=0.25, pos_weight=value),
        "lovasz": lambda: kornia.losses.LovaszSoftmaxLoss(weight=value),
        "rgb": lambda: kornia.color.RgbToRgba(value),
        "bgr": lambda: kornia.color.BgrToRgba(value),
        "alpha": lambda: kornia.enhance.AddWeighted(value, 0.5, 0.1),
        "beta": lambda: kornia.enhance.AddWeighted(0.5, value, 0.1),
        "gamma": lambda: kornia.enhance.AddWeighted(0.5, 0.5, value),
    }
    name = "pos_weight" if case == "binary_pos_weight" else "weight"
    if case in ("rgb", "bgr"):
        name = "alpha_val"
    elif case in ("alpha", "beta", "gamma"):
        name = case
    return constructors[case](), name


CASES = ["dice", "focal", "binary_weight", "binary_pos_weight", "lovasz", "rgb", "bgr", "alpha", "beta", "gamma"]


def _value(case):
    return torch.full((1, 1, 2, 2) if case in ("rgb", "bgr", "alpha", "beta", "gamma") else (3,), 0.7)


def _forward(module, case, device, dtype):
    pred = torch.linspace(-0.7, 0.8, 12, device=device, dtype=dtype).reshape(1, 3, 2, 2)
    if case in ("rgb", "bgr"):
        return module(pred)
    if case in ("alpha", "beta", "gamma"):
        return module(pred, pred * 0.3)
    if case.startswith("binary"):
        return module(pred, torch.ones_like(pred))
    target = torch.tensor([[[0, 1], [2, 1]]], device=device)
    return module(pred, target)


@pytest.mark.parametrize("case", CASES)
def test_constructor_tensor_follows_conversion(case, device, dtype):
    original = _value(case).requires_grad_()
    module, name = _module(case, original)
    module.to(device=device, dtype=dtype)
    converted = getattr(module, name)
    assert converted.device == torch.device(device)
    assert converted.dtype == dtype
    assert name in dict(module.named_buffers())
    assert dict(module.named_parameters()) == {}
    assert dict(module.state_dict()) == {}
    expected, _ = _module(case, original.to(device=device, dtype=dtype))
    actual = _forward(module, case, device, dtype)
    torch.testing.assert_close(actual, _forward(expected, case, device, dtype))
    actual.sum().backward()
    assert original.grad is not None
    assert torch.isfinite(original.grad).all()


@pytest.mark.parametrize("case", CASES)
def test_explicit_parameter_remains_trainable(case, device, dtype):
    parameter = torch.nn.Parameter(_value(case))
    module, name = _module(case, parameter)
    module.to(device=device, dtype=dtype)
    assert getattr(module, name) is parameter
    assert parameter.device == torch.device(device)
    assert parameter.dtype == dtype
    assert dict(module.named_parameters())[name] is parameter
    assert name in module.state_dict()
    _forward(module, case, device, dtype).sum().backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("case", CASES)
def test_constructor_tensor_moves_to_meta(case):
    module, name = _module(case, _value(case))
    module.to(device="meta")
    assert getattr(module, name).device == torch.device("meta")
    if case in ("rgb", "bgr", "alpha", "beta", "gamma"):
        assert _forward(module, case, "meta", torch.float32).device == torch.device("meta")
