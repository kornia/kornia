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

"""Utilities for automating parametrized test generation across devices and dtypes.

This module provides decorators and utilities to automatically generate common test methods
(test_smoke, test_cardinality, test_gradcheck) parametrized across devices and dtypes.

Example:
    >>> @parametrized_test(
    ...     smoke_inputs=lambda device, dtype: (tensor([1.0], device=device, dtype=dtype),),
    ...     cardinality_tests=[
    ...         {"inputs": lambda device, dtype: (tensor([1.0], device=device, dtype=dtype),),
    ...          "expected_shape": (1,)}
    ...     ],
    ... )
    ... class TestMyFunction(BaseTester):
    ...     def setup_method(self):
    ...         self.func = my_function
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import torch

from kornia.core import Dtype, Tensor

from .base import BaseTester


def parametrized_test(
    smoke_inputs: Optional[Callable[[torch.device, Dtype], tuple[Any, ...]]] = None,
    cardinality_tests: Optional[list[dict[str, Any]]] = None,
    gradcheck_inputs: Optional[Callable[[torch.device], tuple[Any, ...]]] = None,
) -> Callable:
    """Decorator to automatically generate parametrized test methods.

    Generates test_smoke, test_cardinality, and test_gradcheck methods that are
    automatically parametrized across devices and dtypes.

    Args:
        smoke_inputs: Callable that takes (device, dtype) and returns input arguments tuple
            for smoke testing. If provided, generates test_smoke method.
        cardinality_tests: List of dicts with 'inputs' (callable) and 'expected_shape' keys.
            'inputs' callable takes (device, dtype) and returns inputs tuple.
            If provided, generates test_cardinality method.
        gradcheck_inputs: Callable that takes (device,) and returns input arguments tuple
            for gradcheck. If provided, generates test_gradcheck method.

    Returns:
        Decorator function that adds parametrized test methods to the test class.

    Example:
        >>> @parametrized_test(
        ...     smoke_inputs=lambda dev, dtype: (torch.randn(2, 3, device=dev, dtype=dtype),),
        ...     cardinality_tests=[
        ...         {"inputs": lambda dev, dtype: (torch.randn(2, 3, device=dev, dtype=dtype),),
        ...          "expected_shape": torch.Size([2, 3])}
        ...     ],
        ...     gradcheck_inputs=lambda dev: (torch.randn(2, 3, device=dev, requires_grad=True, dtype=torch.float64),),
        ... )
        ... class MyTestClass(BaseTester):
        ...     def setup_method(self):
        ...         self.func = some_function
    """

    def decorator(cls: type) -> type:
        # Check if class has a func or function_under_test attribute
        if not hasattr(cls, "func") and not hasattr(cls, "function_under_test"):

            def setup_method_wrapper(self):
                """Default setup that expects subclass to define func or function_under_test."""
                if not hasattr(self, "func") and not hasattr(self, "function_under_test"):
                    raise NotImplementedError(
                        f"{cls.__name__} must define either 'func' or 'function_under_test' attribute or override setup_method()"
                    )

            if not hasattr(cls, "setup_method"):
                cls.setup_method = setup_method_wrapper

        # Generate test_smoke if smoke_inputs provided
        if smoke_inputs is not None:

            def test_smoke(self, device: torch.device, dtype: Dtype) -> None:
                """Smoke test: verify function runs with provided inputs."""
                func = getattr(self, "func", None) or getattr(self, "function_under_test", None)
                if func is None:
                    raise NotImplementedError(f"{cls.__name__} must define 'func' or 'function_under_test'")

                inputs = smoke_inputs(device, dtype)
                try:
                    func(*inputs)
                except Exception as e:
                    raise AssertionError(f"Smoke test failed: {e}") from e

            cls.test_smoke = test_smoke

        # Generate test_cardinality if cardinality_tests provided
        if cardinality_tests is not None:

            def test_cardinality(self, device: torch.device, dtype: Dtype) -> None:
                """Cardinality test: verify output shape matches expected shape."""
                func = getattr(self, "func", None) or getattr(self, "function_under_test", None)
                if func is None:
                    raise NotImplementedError(f"{cls.__name__} must define 'func' or 'function_under_test'")

                for i, test_case in enumerate(cardinality_tests):
                    inputs = test_case["inputs"](device, dtype)
                    expected_shape = test_case["expected_shape"]

                    try:
                        output = func(*inputs)
                    except Exception as e:
                        raise AssertionError(f"Cardinality test {i} failed to execute: {e}") from e

                    if isinstance(output, Tensor):
                        actual_shape = output.shape
                        assert (
                            actual_shape == expected_shape
                        ), f"Test case {i}: Expected shape {expected_shape}, got {actual_shape}"
                    elif isinstance(output, (tuple, list)):
                        for j, out in enumerate(output):
                            if isinstance(out, Tensor):
                                actual_shape = out.shape
                                expected = (
                                    expected_shape[j]
                                    if isinstance(expected_shape, (tuple, list))
                                    else expected_shape
                                )
                                assert (
                                    actual_shape == expected
                                ), f"Test case {i}, output {j}: Expected shape {expected}, got {actual_shape}"

            cls.test_cardinality = test_cardinality

        # Generate test_gradcheck if gradcheck_inputs provided
        if gradcheck_inputs is not None:

            def test_gradcheck(self, device: torch.device) -> None:
                """Gradcheck test: verify gradient computation."""
                func = getattr(self, "func", None) or getattr(self, "function_under_test", None)
                if func is None:
                    raise NotImplementedError(f"{cls.__name__} must define 'func' or 'function_under_test'")

                inputs = gradcheck_inputs(device)
                try:
                    result = self.gradcheck(func, inputs)
                    assert result, "Gradcheck failed"
                except Exception as e:
                    raise AssertionError(f"Gradcheck test failed: {e}") from e

            cls.test_gradcheck = test_gradcheck

        return cls

    return decorator
