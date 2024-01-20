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
    return tensor.requires_grad_(requires_grad).type(dtype)


class BaseTester:
    @staticmethod
    def assert_close(
        actual: Tensor,
        expected: Tensor,
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

        if "xla" in actual.device.type or "xla" in expected.device.type:
            rtol, atol = 1e-2, 1e-2

        if rtol is None and atol is None:
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
        **kwargs: Any,
    ) -> bool:
        if isinstance(inputs, torch.Tensor):
            inputs = tensor_to_gradcheck_var(inputs)
        else:
            inputs = [tensor_to_gradcheck_var(i) if isinstance(i, torch.Tensor) else i for i in inputs]

        return gradcheck(func, inputs, raise_exception=raise_exception, fast_mode=fast_mode, **kwargs)
