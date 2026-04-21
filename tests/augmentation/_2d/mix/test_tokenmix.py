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
    params = aug.generate_parameters(torch.Size([batch_size, channels, height, width]))
    out = aug.apply_transform(x, params, {})
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_tokenmix_params_keys():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    assert "lam" in params
    assert "num_mix_tokens" in params
    assert "batch_perm" in params
    assert params["lam"].shape[0] == 4
    assert params["batch_perm"].shape[0] == 4


def test_tokenmix_lam_controls_mixing():
    """Images with lam→0 (0 tokens mixed) should be closer to original than lam→1."""
    aug = TokenMix(num_tokens=4)
    x = torch.rand(2, 3, 32, 32)
    params_none = aug.generate_parameters(x.shape)
    params_none["lam"] = torch.zeros(2)
    params_none["num_mix_tokens"] = torch.zeros(2, dtype=torch.long)
    out_none = aug.apply_transform(x, params_none, {})
    assert torch.allclose(out_none, x), "With 0 tokens mixed, output should equal input"


def test_tokenmix_different_inputs():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    out1 = aug.apply_transform(x, params, {})
    params2 = aug.generate_parameters(y.shape)
    out2 = aug.apply_transform(y, params2, {})
    assert not torch.allclose(out1, out2)


def test_tokenmix_grad():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    params = aug.generate_parameters(x.shape)
    out = aug.apply_transform(x, params, {})
    out.sum().backward()
    assert x.grad is not None


def test_tokenmix_invalid_size():
    aug = TokenMix(alpha=1.0, num_tokens=5)
    x = torch.rand(2, 3, 32, 32)  # 32 not divisible by 5
    params = aug.generate_parameters(x.shape)
    with pytest.raises(ValueError, match="divisible"):
        aug.apply_transform(x, params, {})


def test_tokenmix_num_tokens_too_large():
    aug = TokenMix(alpha=1.0, num_tokens=64)
    x = torch.rand(2, 3, 16, 16)  # num_tokens > H, token_h = 0
    params = aug.generate_parameters(x.shape)
    with pytest.raises(ValueError, match="exceeds"):
        aug.apply_transform(x, params, {})
