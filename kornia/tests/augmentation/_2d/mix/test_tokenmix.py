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

# ruff: noqa: D103, S101
import pytest
import torch

pytest.importorskip("kornia.augmentation._2d.mix.tokenmix")
from kornia.augmentation._2d.mix.tokenmix import TokenMix


@pytest.mark.parametrize(
    "batch_size,channels,height,width,num_tokens",
    [
        (4, 3, 32, 32, 4),
        (2, 1, 16, 16, 2),
    ],
)
def test_tokenmix_shape_and_type(batch_size, channels, height, width, num_tokens):
    aug = TokenMix(alpha=1.0, num_tokens=num_tokens)
    x = torch.rand(batch_size, channels, height, width)
    params = aug.generate_parameters(x.shape)
    out, idx, lam = aug.apply_transform(x, params, {})
    assert out.shape == x.shape
    assert idx.shape[0] == batch_size
    assert lam.shape[0] == batch_size
    assert out.dtype == x.dtype


def test_tokenmix_different_inputs():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    out1, _, _ = aug.apply_transform(x, params, {})
    out2, _, _ = aug.apply_transform(y, params, {})
    assert not torch.allclose(out1, out2)


def test_tokenmix_grad():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    params = aug.generate_parameters(x.shape)
    out, _, _ = aug.apply_transform(x, params, {})
    out.sum().backward()
    assert x.grad is not None
