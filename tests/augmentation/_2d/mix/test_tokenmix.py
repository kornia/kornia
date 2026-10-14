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


def test_tokenmix_onnx_exportable():
    aug = TokenMix(num_tokens=4)
    assert not aug.exportable
    assert not TokenMix.ONNX_EXPORTABLE


def test_tokenmix_spatial_validation_before_gate():
    # Spatial checks must fail regardless of probability gate (p=0, p=0.5, p=1.0)
    for p_val in [0.0, 0.5, 1.0]:
        aug = TokenMix(num_tokens=5, p=p_val)
        x_invalid = torch.rand(2, 1, 16, 16)
        with pytest.raises(ValueError, match="divisible"):
            aug(x_invalid)

    # Spatial check fails in generate_parameters
    aug = TokenMix(num_tokens=5, p=1.0)
    with pytest.raises(ValueError, match="divisible"):
        aug.generate_parameters(torch.Size([2, 1, 16, 16]))

    # num_tokens exceeds spatial dimension
    aug_large = TokenMix(num_tokens=64, p=0.0)
    with pytest.raises(ValueError, match="exceeds"):
        aug_large(torch.rand(2, 1, 16, 16))

    # Replay with supplied parameters whose gate is disabled (batch_prob=0)
    aug_valid = TokenMix(num_tokens=4, p=1.0)
    valid_params = aug_valid.forward_parameters(torch.Size([2, 1, 16, 16]))
    valid_params["batch_prob"] = torch.zeros(2)
    # Applying invalid spatial size with disabled gate must still fail
    x_bad = torch.rand(2, 1, 18, 18)
    with pytest.raises(ValueError, match="divisible"):
        aug_valid(x_bad, params=valid_params)


def test_tokenmix_rng_device_and_dtype_migration():
    # float64 dtype migration
    aug = TokenMix(num_tokens=2).to(dtype=torch.float64)
    params = aug.forward_parameters(torch.Size([2, 1, 4, 4]))
    assert params["lam"].dtype == torch.float64
    assert params["batch_prob"].dtype == torch.float64
    assert params["token_mask"].dtype == torch.bool
    assert params["lam"].device == torch.device("cpu")
    assert params["batch_perm"].device == torch.device("cpu")
    assert params["token_mask"].device == torch.device("cpu")

    # set_rng_device_and_dtype
    aug.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
    params_f32 = aug.forward_parameters(torch.Size([2, 1, 4, 4]))
    assert params_f32["lam"].dtype == torch.float32

    # Empty batch respects device and dtype
    aug_empty = TokenMix(num_tokens=2).to(dtype=torch.float64)
    params_empty = aug_empty.generate_parameters(torch.Size([0, 1, 4, 4]))
    assert params_empty["lam"].dtype == torch.float64
    assert params_empty["lam"].device == torch.device("cpu")
    assert params_empty["num_mix_tokens"].device == torch.device("cpu")
    assert params_empty["batch_perm"].device == torch.device("cpu")
    assert params_empty["token_mask"].device == torch.device("cpu")

    # Reseeding reproducibility
    torch.manual_seed(123)
    aug_seed1 = TokenMix(num_tokens=2)
    p_seed1 = aug_seed1.forward_parameters(torch.Size([2, 1, 4, 4]))
    torch.manual_seed(123)
    aug_seed2 = TokenMix(num_tokens=2)
    p_seed2 = aug_seed2.forward_parameters(torch.Size([2, 1, 4, 4]))
    assert torch.equal(p_seed1["lam"], p_seed2["lam"])
    assert torch.equal(p_seed1["batch_perm"], p_seed2["batch_perm"])
    assert torch.equal(p_seed1["token_mask"], p_seed2["token_mask"])

    # Optional accelerator device migration (CUDA or MPS if present)
    dev_names = []
    if torch.cuda.is_available():
        dev_names.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev_names.append("mps")
    for dev_name in dev_names:
        dev = torch.device(dev_name)
        aug_dev = TokenMix(num_tokens=2).to(device=dev)
        p_dev = aug_dev.forward_parameters(torch.Size([2, 1, 4, 4]))
        assert p_dev["lam"].device.type == dev.type
        assert p_dev["batch_perm"].device.type == dev.type
        assert p_dev["token_mask"].device.type == dev.type
        assert p_dev["num_mix_tokens"].device.type == dev.type
