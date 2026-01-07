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

import warnings
from typing import Any, Callable, Optional, Union

import torch

Dtype = Union[torch.dtype, None]
Tensor = torch.Tensor


def parametrized_test(
    smoke_inputs: Optional[Callable[[torch.device, Dtype], tuple[Any, ...]]] = None,
    cardinality_tests: Optional[list[dict[str, Any]]] = None,
    gradcheck_inputs: Optional[Callable[[torch.device], tuple[Any, ...]]] = None,
) -> Callable[[type], type]:
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
                        f"{cls.__name__} must define either 'func' or 'function_under_test' "
                        "attribute or override setup_method()"
                    )

            if not hasattr(cls, "setup_method"):
                cls.setup_method = setup_method_wrapper

        _generate_test_smoke(cls, smoke_inputs)
        _generate_test_cardinality(cls, cardinality_tests)
        _generate_test_gradcheck(cls, gradcheck_inputs)

        return cls

    return decorator


def _generate_test_smoke(
    cls: type,
    smoke_inputs: Optional[Callable[[torch.device, Dtype], tuple[Any, ...]]] = None,
) -> None:
    """Generate test_smoke method if smoke_inputs provided."""
    if smoke_inputs is None:
        return

    # Warn if the method already exists (but don't fail, as parent classes may define it)
    if "test_smoke" in cls.__dict__:
        warnings.warn(
            f"{cls.__name__} already defines 'test_smoke' method. "
            "The @parametrized_test decorator will overwrite it. "
            "Remove the existing method or the decorator parameter.",
            UserWarning,
            stacklevel=2,
        )

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


def _generate_test_cardinality(
    cls: type,
    cardinality_tests: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Generate test_cardinality method if cardinality_tests provided."""
    if cardinality_tests is None:
        return

    # Warn if the method already exists (but don't fail, as parent classes may define it)
    if "test_cardinality" in cls.__dict__:
        warnings.warn(
            f"{cls.__name__} already defines 'test_cardinality' method. "
            "The @parametrized_test decorator will overwrite it. "
            "Remove the existing method or the decorator parameter.",
            UserWarning,
            stacklevel=2,
        )

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

            _check_output_shape(cls, output, expected_shape, i)

    cls.test_cardinality = test_cardinality


def _check_output_shape(
    cls: type,
    output: Any,
    expected_shape: Any,
    test_case_idx: int,
) -> None:
    """Check if output shape matches expected shape."""
    if isinstance(output, Tensor):
        actual_shape = output.shape
        assert actual_shape == expected_shape, (
            f"Test case {test_case_idx}: Expected shape {expected_shape}, got {actual_shape}"
        )
    elif isinstance(output, (tuple, list)):
        for j, out in enumerate(output):
            if isinstance(out, Tensor):
                actual_shape = out.shape
                expected = expected_shape[j] if isinstance(expected_shape, (tuple, list)) else expected_shape
                assert actual_shape == expected, (
                    f"Test case {test_case_idx}, output {j}: Expected shape {expected}, got {actual_shape}"
                )


def _generate_test_gradcheck(
    cls: type,
    gradcheck_inputs: Optional[Callable[[torch.device], tuple[Any, ...]]] = None,
) -> None:
    """Generate test_gradcheck method if gradcheck_inputs provided."""
    if gradcheck_inputs is None:
        return

    # Warn if the method already exists (but don't fail, as parent classes may define it)
    if "test_gradcheck" in cls.__dict__:
        warnings.warn(
            f"{cls.__name__} already defines 'test_gradcheck' method. "
            "The @parametrized_test decorator will overwrite it. "
            "Remove the existing method or the decorator parameter.",
            UserWarning,
            stacklevel=2,
        )

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
