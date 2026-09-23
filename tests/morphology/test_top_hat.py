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

from kornia.morphology import opening, top_hat

from testing.base import BaseTester, assert_close
from testing.parametrized_tester import parametrized_test


@parametrized_test(
    smoke_inputs=lambda device, dtype: (
        torch.rand(1, 3, 4, 4, device=device, dtype=dtype),
        torch.ones((3, 3), device=device, dtype=dtype),
    ),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (
                torch.ones((1, 3, 4, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([1, 3, 4, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((2, 3, 2, 4), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([2, 3, 2, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 3, 4, 1), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 3, 4, 1]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.ones((3, 2, 5, 5), device=device, dtype=dtype),
                torch.ones((3, 3), device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 2, 5, 5]),
        },
    ],
    gradcheck_inputs=lambda device: (
        torch.rand(2, 3, 4, 4, requires_grad=True, device=device, dtype=torch.float64),
        torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64),
    ),
)
class TestTopHat(BaseTester):
    def setup_method(self) -> None:
        self.func = top_hat

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 0.5, 0.0], [0.2, 0.0, 0.5], [0.0, 0.5, 0.0]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(top_hat(tensor, kernel), expected)

    def test_structural_element(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        structural_element = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, -1.0]], device=device, dtype=dtype
        )
        expected = torch.tensor([[0.0, 0.5, 0.0], [0.2, 0.0, 0.5], [0.0, 0.5, 0.0]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        assert_close(
            top_hat(tensor, torch.ones_like(structural_element), structuring_element=structural_element),
            expected,
        )

    def test_exception(self, device, dtype):
        sample = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            assert top_hat([0.0], kernel)

        with pytest.raises(TypeError):
            assert top_hat(sample, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert top_hat(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(2, 3, 4, device=device, dtype=dtype)
            assert top_hat(sample, test)

    def test_jit(self, device, dtype):
        op = top_hat
        op_script = torch.jit.script(op)

        sample = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(sample, kernel)
        expected = op(sample, kernel)

        assert_close(actual, expected)

    def test_convention_top_hat_is_image_minus_opening(self, device, dtype):
        # `top_hat` is exactly `x - opening(x)` with the same kernel and the same options, so every
        # convention of :func:`kornia.morphology.opening` applies to it unchanged. This pins the
        # composition: both sides call the same `opening`, whose conventions are pinned in
        # test_opening.py. The equality is repeated with a non-default `border_type`, `border_value` and
        # `origin`, so a `top_hat` that dropped one of the options would show.
        # `top_hat` evaluates that very expression, so the two sides are bitwise equal in every dtype.
        # Generated with:
        #   L = torch.tensor([[0., 0., 0.], [0., 1., 1.], [0., 1., 0.]])
        #   torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0))
        # A local `torch.Generator` avoids touching the process-global (and any device) RNG state.
        l_kernel = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        tensor = torch.rand(1, 1, 7, 10, generator=torch.Generator().manual_seed(0)).to(device=device, dtype=dtype)

        assert torch.equal(top_hat(tensor, l_kernel), tensor - opening(tensor, l_kernel))
        # The opening is anti-extensive, so the top hat is non-negative.
        assert (top_hat(tensor, l_kernel) >= 0).all()
        options = {"border_type": "constant", "border_value": 0.5, "origin": [0, 0]}
        assert torch.equal(top_hat(tensor, l_kernel, **options), tensor - opening(tensor, l_kernel, **options))
