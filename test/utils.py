import contextlib
from typing import Any, Optional

import torch

__all__ = ["assert_close"]

try:
    # torch.testing.assert_close is only available for torch>=1.9
    from torch.testing import assert_close as _assert_close
    from torch.testing._core import _get_default_tolerance

    def assert_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if rtol is None and atol is None:
            with contextlib.suppress(Exception):
                rtol, atol = _get_default_tolerance(actual, expected)

        return _assert_close(actual, expected, rtol=rtol, atol=atol, check_stride=False, equal_nan=True, **kwargs)


except ImportError:
    # Partial backport of torch.testing.assert_close for torch<1.9
    from torch.testing import assert_allclose as _assert_allclose

    class UsageError(Exception):
        pass

    def assert_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        try:
            return _assert_allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)
        except ValueError as error:
            raise UsageError(str(error)) from error
