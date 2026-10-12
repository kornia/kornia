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

from __future__ import annotations

import pytest
import torch

from kornia.augmentation import TokenMix

from testing.base import BaseTester


class TestTokenMixPublicAPI(BaseTester):
    def test_batch_probability(self, device, dtype):
        x = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)

        # p=0: never applied, output identical to input
        aug_0 = TokenMix(alpha=1.0, num_tokens=4, p=0.0)
        out_0 = aug_0(x)
        self.assert_close(out_0, x)

        # p=1: always applied
        aug_1 = TokenMix(alpha=1.0, num_tokens=4, p=1.0)
        out_1 = aug_1(x)
        assert out_1.shape == x.shape
        assert bool(aug_1._params["batch_prob"].all())

        # p=0.5: single Bernoulli gate across 16 seeds
        for seed in range(16):
            for same_on_batch in [False, True]:
                torch.manual_seed(seed)
                aug = TokenMix(alpha=1.0, num_tokens=4, p=0.5, same_on_batch=same_on_batch)
                out = aug(x)
                bp = aug._params["batch_prob"]
                assert (bp > 0.5).all() or (bp < 0.5).all()
                if (bp < 0.5).all():
                    self.assert_close(out, x)
                # Replay verification
                out_replayed = aug(x, params=aug._params)
                self.assert_close(out, out_replayed)

    def test_shared_strength_and_partner_semantics(self, device, dtype):
        x = torch.rand(4, 3, 32, 32, device=device, dtype=dtype)

        # same_on_batch=True: shared lam and token count across batch
        aug_shared = TokenMix(alpha=1.0, num_tokens=4, p=1.0, same_on_batch=True)
        out_shared = aug_shared(x)
        params_shared = aug_shared._params
        assert (params_shared["num_mix_tokens"] == params_shared["num_mix_tokens"][0]).all()
        assert (params_shared["lam"] == params_shared["lam"][0]).all()

        # Valid partner permutation (a permutation of 0..B-1 without self-pairing for B > 1)
        perm = params_shared["batch_perm"]
        assert torch.equal(torch.sort(perm).values, torch.arange(4, device=perm.device))
        assert not (perm == torch.arange(4, device=perm.device)).any()

        # Shared token mask across batch when same_on_batch=True
        assert "token_mask" in params_shared
        assert torch.equal(params_shared["token_mask"], params_shared["token_mask"][0].repeat(4, 1, 1))

        # Independent sampling when same_on_batch=False
        torch.manual_seed(42)
        aug_indep = TokenMix(alpha=1.0, num_tokens=4, p=1.0, same_on_batch=False)
        aug_indep(x)
        params_indep = aug_indep._params
        assert params_indep["num_mix_tokens"].shape == torch.Size([4])

        # Replay without consuming RNG
        state_before = torch.get_rng_state()
        out_replayed = aug_shared(x, params=params_shared)
        state_after = torch.get_rng_state()
        assert torch.equal(state_before, state_after)
        self.assert_close(out_shared, out_replayed)

    def test_empty_singleton_and_keepdim(self, device, dtype):
        aug = TokenMix(alpha=1.0, num_tokens=4, p=1.0)

        # Empty batch (0, 3, 32, 32)
        x_empty = torch.empty(0, 3, 32, 32, device=device, dtype=dtype)
        out_empty = aug(x_empty)
        assert out_empty.shape == x_empty.shape
        assert out_empty.device == x_empty.device
        assert out_empty.dtype == x_empty.dtype

        # Singleton batch (1, 3, 32, 32) - paired with itself, output == input
        x_single = torch.rand(1, 3, 32, 32, device=device, dtype=dtype)
        out_single = aug(x_single)
        assert out_single.shape == x_single.shape
        self.assert_close(out_single, x_single)

        # Unbatched input with keepdim=True (3, 32, 32)
        aug_keepdim = TokenMix(alpha=1.0, num_tokens=4, p=1.0, keepdim=True)
        x_unbatched = torch.rand(3, 32, 32, device=device, dtype=dtype)
        out_unbatched = aug_keepdim(x_unbatched)
        assert out_unbatched.shape == x_unbatched.shape
        assert out_unbatched.device == x_unbatched.device
        assert out_unbatched.dtype == x_unbatched.dtype

    def test_gradient_with_replay(self, device, dtype):
        if dtype not in (torch.float32, torch.float64):
            pytest.skip("Gradient check requires float32 or float64")
        aug = TokenMix(alpha=1.0, num_tokens=4, p=1.0)
        x = torch.rand(2, 3, 16, 16, device=device, dtype=dtype, requires_grad=True)
        out = aug(x)
        assert out.shape == x.shape
        params = aug._params
        out_replayed = aug(x, params=params)
        out_replayed.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.parametrize(
    "batch_size,channels,height,width,num_tokens",
    [
        (4, 3, 32, 32, 4),
        (2, 1, 16, 16, 2),
    ],
)
def test_tokenmix_shape_and_type(batch_size, channels, height, width, num_tokens):
    aug = TokenMix(alpha=1.0, num_tokens=num_tokens, p=1.0)
    x = torch.rand(batch_size, channels, height, width)
    out = aug(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_tokenmix_params_keys():
    aug = TokenMix(alpha=1.0, num_tokens=4)
    x = torch.rand(4, 3, 32, 32)
    params = aug.generate_parameters(x.shape)
    assert "lam" in params
    assert "num_mix_tokens" in params
    assert "batch_perm" in params
    assert "token_mask" in params
    assert params["lam"].shape[0] == 4
    assert params["batch_perm"].shape[0] == 4
    assert params["token_mask"].shape == (4, 4, 4)


def test_tokenmix_different_inputs():
    aug = TokenMix(alpha=1.0, num_tokens=4, p=1.0)
    x = torch.rand(4, 3, 32, 32)
    y = torch.rand(4, 3, 32, 32)
    out1 = aug(x)
    out2 = aug(y, params=aug._params)
    assert not torch.allclose(out1, out2)


def test_tokenmix_grad():
    aug = TokenMix(alpha=1.0, num_tokens=4, p=1.0)
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    out = aug(x)
    out.sum().backward()
    assert x.grad is not None


def test_tokenmix_invalid_size():
    aug = TokenMix(alpha=1.0, num_tokens=5, p=1.0)
    x = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError, match="divisible"):
        aug(x)


def test_tokenmix_num_tokens_too_large():
    aug = TokenMix(alpha=1.0, num_tokens=64, p=1.0)
    x = torch.rand(2, 3, 16, 16)
    with pytest.raises(ValueError, match="exceeds"):
        aug(x)
