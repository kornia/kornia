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
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend

from kornia.augmentation import RandomClahe

from testing.base import BaseTester


class TestClaheCompile(BaseTester):
    @pytest.mark.parametrize("differentiable", [False, True])
    @pytest.mark.parametrize("backend", ["eager", "inductor"])
    def test_compile_replay(self, device, dtype, torch_optimizer, differentiable, backend):
        if backend == "inductor" and dtype in (torch.float16, torch.bfloat16):
            pytest.skip("Inductor changes half-precision lookup rounding in scalar CLAHE as well")
        if differentiable and backend == "inductor" and device.type == "cpu" and torch.__version__.startswith("2.5.1"):
            pytest.skip(
                "PyTorch 2.5.1 CPU Inductor emits invalid VecMask code for differentiable CLAHE, including scalar CLAHE"
            )
        aug = RandomClahe(grid_size=(2, 3), slow_and_differentiable=differentiable, p=1)
        input = torch.rand(2, 2, 12, 18, device=device, dtype=dtype, requires_grad=differentiable)
        params = aug.forward_parameters(input.shape)
        counter = CompileCounter() if backend == "eager" else CompileCounterWithBackend(backend)
        fn = torch_optimizer(aug, backend=counter, fullgraph=True)
        for limits in [(2.0, 40.0), (0.0, 3.0), (-1.0, 0.0), (7.0, 7.0), (40.0, 2.0), (256 / 36, 512 / 36)]:
            params["clip_limit_factor"] = torch.tensor(limits, device=device, dtype=dtype)
            expected = aug(input, params=params)
            actual = fn(input, params=params)
            self.assert_close(actual, expected)
            if differentiable:
                weights = torch.rand_like(actual)
                actual_grad = torch.autograd.grad(actual, input, weights)[0]
                expected_grad = torch.autograd.grad(expected, input, weights)[0]
                if backend == "eager":
                    self.assert_close(actual_grad, expected_grad)
                else:
                    # Inductor also changes the scalar API's Gaussian-histogram gradients
                    # slightly; the counter backend above isolates per-image clipping parity.
                    assert torch.isfinite(actual_grad).all()
        assert counter.frame_count == 1

    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_compile_random_backend(self, device, dtype, torch_optimizer, same_on_batch):
        aug = RandomClahe(clip_limit=(2.0, 40.0), grid_size=(2, 2), same_on_batch=same_on_batch, p=1)
        # A concentrated histogram responds to every clipping threshold; uniform noise can
        # remain unchanged across many draws because its bins never reach the sampled limits.
        input = torch.zeros(2, 1, 16, 16, device=device, dtype=dtype)
        counter = CompileCounterWithBackend("inductor")
        fn = torch_optimizer(aug, backend=counter, fullgraph=True)
        outputs = [fn(input) for _ in range(12)]
        assert all(output.shape == input.shape and output.dtype == input.dtype for output in outputs)
        assert all(torch.isfinite(output).all() for output in outputs)
        assert any(not torch.equal(outputs[0], output) for output in outputs[1:])
        assert counter.frame_count == 1

    def test_compile_backend_gradients(self, device, dtype, torch_optimizer):
        if device.type == "cpu" and torch.__version__.startswith("2.5.1"):
            pytest.skip(
                "PyTorch 2.5.1 CPU Inductor emits invalid VecMask code for differentiable CLAHE, including scalar CLAHE"
            )
        # At the first bin, linspace rounding does not amplify differences through the narrow
        # Gaussian kernel, so the real backend's gradients can be compared at normal tolerances.
        aug = RandomClahe(grid_size=(2, 3), slow_and_differentiable=True, p=1)
        input = torch.zeros(2, 2, 12, 18, device=device, dtype=dtype, requires_grad=True)
        params = aug.forward_parameters(input.shape)
        params["clip_limit_factor"] = torch.tensor([0.0, 40.0], device=device, dtype=dtype)
        actual = torch_optimizer(aug, fullgraph=True)(input, params=params)
        expected = aug(input, params=params)
        self.assert_close(actual, expected)
        weights = torch.rand_like(actual)
        actual_grad = torch.autograd.grad(actual, input, weights)[0]
        expected_grad = torch.autograd.grad(expected, input, weights)[0]
        assert expected_grad.abs().max() > 0
        self.assert_close(actual_grad, expected_grad)
