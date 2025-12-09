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

"""Proof of Concept: Parametrized test automation using the new decorator.

This demonstrates the usage of the @parametrized_test decorator on existing
test classes to automatically generate test_smoke, test_cardinality, and
test_gradcheck methods parametrized across devices and dtypes.
"""

import torch

from kornia.core import tensor
from kornia.utils.misc import differentiable_clipping, differentiable_polynomial_floor

from testing.base import BaseTester
from testing.parametrized_tester import parametrized_test


@parametrized_test(
    smoke_inputs=lambda device, dtype: (tensor([1.0, 6.0, 10.0, 12.0], device=device, dtype=dtype),),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (tensor([1.0, 6.0, 10.0, 12.0], device=device, dtype=dtype),),
            "expected_shape": torch.Size([4]),
        }
    ],
    gradcheck_inputs=lambda device: (
        tensor([1.0, 6.0, 11.0, 12.0], device=device, dtype=torch.float64, requires_grad=True),
    ),
)
class TestDifferentiableClippingAutomated(BaseTester):
    """Demonstration of automated tests for differentiable_clipping function.

    The @parametrized_test decorator automatically generates:
    - test_smoke: Runs the function with valid inputs
    - test_cardinality: Verifies output shapes
    - test_gradcheck: Validates gradient computation

    All tests are parametrized across devices and dtypes automatically.
    """

    def __init__(self):
        self.func = lambda x: differentiable_clipping(x, min_val=5.0, max_val=10.0)


@parametrized_test(
    smoke_inputs=lambda device, dtype: (tensor([1.5, 3.1, 5.9, 6.6], device=device, dtype=dtype),),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (tensor([1.5, 3.1, 5.9, 6.6], device=device, dtype=dtype),),
            "expected_shape": torch.Size([4]),
        }
    ],
    gradcheck_inputs=lambda device: (
        tensor([1.5, 3.1, 5.9, 6.6], device=device, dtype=torch.float64, requires_grad=True),
    ),
)
class TestDifferentiablePolynomialFloorAutomated(BaseTester):
    """Demonstration of automated tests for differentiable_polynomial_floor function.

    Uses the same @parametrized_test decorator pattern as TestDifferentiableClippingAutomated.
    """

    def __init__(self):
        self.func = differentiable_polynomial_floor
