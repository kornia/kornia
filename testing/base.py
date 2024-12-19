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

import math
from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch.autograd import gradcheck
from torch.testing import assert_close as _assert_close

from kornia.core import Dtype, Tensor

# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.bfloat16: (7.8e-3, 7.8e-3),
    torch.float16: (9.7e-4, 9.7e-4),
    torch.float32: (1e-4, 1e-5),  # TODO: Update to ~1.2e-7
    # TODO: Update to ~2.3e-16 for fp64
    torch.float64: (1e-5, 1e-5),  # TODO: BaseTester used (1.3e-6, 1e-5), but it fails for general cases
}


def _default_tolerances(*inputs: Any) -> tuple[float, float]:
    rtols, atols = zip(*[_DTYPE_PRECISIONS.get(torch.as_tensor(input_).dtype, (0.0, 0.0)) for input_ in inputs])
    return max(rtols), max(atols)


def assert_close(
    actual: Tensor, expected: Tensor, *, rtol: Optional[float] = None, atol: Optional[float] = None, **kwargs: Any
) -> None:
    if rtol is None and atol is None:
        # `torch.testing.assert_close` used different default tolerances than `torch.testing.assert_allclose`.
        # TODO: remove this special handling as soon as https://github.com/kornia/kornia/issues/1134 is resolved
        #  Basically, this whole wrapper function can be removed and `torch.testing.assert_close` can be used
        #  directly.
        rtol, atol = _default_tolerances(actual, expected)

    return _assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        # this is the default value for torch>=1.10, but not for torch==1.9
        # TODO: remove this if kornia relies on torch>=1.10
        check_stride=False,
        equal_nan=False,
        **kwargs,
    )


def tensor_to_gradcheck_var(
    tensor: Tensor, dtype: Dtype = torch.float64, requires_grad: bool = True
) -> Union[Tensor, str]:
    """Convert the input tensor to a valid variable to check the gradient.

    `gradcheck` needs 64-bit floating point and requires gradient.
    """
    if not torch.is_tensor(tensor):
        raise AssertionError(type(tensor))
    t = tensor.type(dtype)

    if t.is_floating_point():
        return t.requires_grad_(requires_grad)

    return t


class BaseTester:
    @staticmethod
    def assert_close(
        actual: Tensor | float,
        expected: Tensor | float,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        low_tolerance: bool = False,
    ) -> None:
        """Asserts that `actual` and `expected` are close.

        Args:
            actual: Actual input.
            expected: Expected input.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            low_tolerance:
                This parameter allows to reduce tolerance. Half the decimal places.
                Example, 1e-4 -> 1e-2 or 1e-6 -> 1e-3

        """
        if hasattr(actual, "data"):
            actual = actual.data
        if hasattr(expected, "data"):
            expected = expected.data

        if (isinstance(actual, Tensor) and "xla" in actual.device.type) or (
            isinstance(expected, Tensor) and "xla" in expected.device.type
        ):
            rtol, atol = 1e-2, 1e-2

        if (isinstance(actual, Tensor) and isinstance(expected, Tensor)) and rtol is None and atol is None:
            actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
            expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
            rtol, atol = max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)

            # halve the tolerance if `low_tolerance` is true
            rtol = math.sqrt(rtol) if low_tolerance else rtol
            atol = math.sqrt(atol) if low_tolerance else atol

        return assert_close(actual, expected, rtol=rtol, atol=atol)

    @staticmethod
    def gradcheck(
        func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
        inputs: Union[torch.Tensor, Sequence[Any]],
        *,
        raise_exception: bool = True,
        fast_mode: bool = True,
        requires_grad: Sequence[bool] = [],
        dtypes: Sequence[Dtype] = [],
        **kwargs: Any,
    ) -> bool:
        """It will gradcheck the function using the `torch.autograd.gradcheck` method.

        By default this method will pass all tensor to `tensor_to_gradcheck_var` which casts the tensor
        to be float64 dtype, and requires grad as True. You can overwrite which tensors should have requires grad
        equals True, by using a Sequence of the same length of the sequence of inputs, within the requires_grad
        per item. You also, can overwrite with the same mechanics the dtype using the `dtypes`
        parameter.
        """
        requires_grad = requires_grad if len(requires_grad) > 0 else [True] * len(inputs)
        dtypes = dtypes if len(dtypes) > 0 else [torch.float64] * len(inputs)

        if isinstance(inputs, torch.Tensor):
            inputs = tensor_to_gradcheck_var(inputs)
        elif isinstance(inputs, dict):
            inputs = {
                k: tensor_to_gradcheck_var(v, d, r) if isinstance(v, torch.Tensor) else v
                for (k, v), d, r in zip(inputs.items(), dtypes, requires_grad)
            }
        else:
            inputs = [
                tensor_to_gradcheck_var(i, d, r) if isinstance(i, torch.Tensor) else i
                for i, r, d in zip(inputs, requires_grad, dtypes)
            ]

        return gradcheck(func, inputs, raise_exception=raise_exception, fast_mode=fast_mode, **kwargs)
