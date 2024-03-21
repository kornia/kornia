"""The testing package contains testing-specific utilities."""

import importlib.util
import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, TypeVar, Union

import torch
from torch.autograd import gradcheck
from torch.testing import assert_close as _assert_close

from kornia.core import Device, Dtype, Tensor, eye, tensor
from kornia.utils.helpers import deprecated

warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    (
        "Since kornia 0.7.2 the `kornia.testing` module is deprecated and will be removed in kornia 0.8.0 (dec 2024)."
        " Most of these functionalities will be removed from kornia package and will be part of the tests of kornia."
        " Some functionalities which we think is important to keep will be moved to other kornia module."
    ),
    category=DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", DeprecationWarning)

__all__ = ["tensor_to_gradcheck_var", "create_eye_batch", "xla_is_available", "assert_close"]


@deprecated(
    "kornia.utils.xla_is_available",
    "0.7.2",
    extra_reason="The `kornia.testing` module is deprecated and will be removed in kornia 0.8.0 (dec 2024).",
)
def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


@deprecated(
    "kornia.utils.is_mps_tensor_safe",
    "0.7.2",
    extra_reason="The `kornia.testing` module is deprecated and will be removed in kornia 0.8.0 (dec 2024).",
)
def is_mps_tensor_safe(x: Tensor) -> bool:
    """Return whether tensor is on MPS device."""
    return "mps" in str(x.device)


@deprecated(
    "kornia.utils.misc.eye_like",
    "0.7.2",
    extra_reason="The `kornia.testing` module is deprecated and will be removed in kornia 0.8.0 (dec 2024).",
)
def create_eye_batch(batch_size: int, eye_size: int, device: Device = None, dtype: Dtype = None) -> Tensor:
    """Create a batch of identity matrices of shape Bx3x3."""
    return eye(eye_size, device=device, dtype=dtype).view(1, eye_size, eye_size).expand(batch_size, -1, -1)


def create_random_homography(batch_size: int, eye_size: int, std_val: float = 1e-3) -> Tensor:
    """Create a batch of random homographies of shape Bx3x3."""
    std = torch.FloatTensor(batch_size, eye_size, eye_size)
    eye = create_eye_batch(batch_size, eye_size)
    return eye + std.uniform_(-std_val, std_val)


def tensor_to_gradcheck_var(
    tensor: Tensor, dtype: Dtype = torch.float64, requires_grad: bool = True
) -> Union[Tensor, str]:
    """Convert the input tensor to a valid variable to check the gradient.

    `gradcheck` needs 64-bit floating point and requires gradient.
    """
    if not torch.is_tensor(tensor):
        raise AssertionError(type(tensor))
    return tensor.requires_grad_(requires_grad).type(dtype)


T = TypeVar("T")


def dict_to(data: Dict[T, Any], device: Device, dtype: Dtype) -> Dict[T, Any]:
    out: Dict[T, Any] = {}
    for key, val in data.items():
        out[key] = val.to(device, dtype) if isinstance(val, Tensor) else val
    return out


def compute_patch_error(x: Tensor, y: Tensor, h: int, w: int) -> Tensor:
    """Compute the absolute error between patches."""
    return torch.abs(x - y)[..., h // 4 : -h // 4, w // 4 : -w // 4].mean()


def create_rectified_fundamental_matrix(batch_size: int) -> Tensor:
    """Create a batch of rectified fundamental matrices of shape Bx3x3."""
    F_rect = tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).view(1, 3, 3)
    F_repeat = F_rect.expand(batch_size, 3, 3)
    return F_repeat


def create_random_fundamental_matrix(batch_size: int, std_val: float = 1e-3) -> Tensor:
    """Create a batch of random fundamental matrices of shape Bx3x3."""
    F_rect = create_rectified_fundamental_matrix(batch_size)
    H_left = create_random_homography(batch_size, 3, std_val)
    H_right = create_random_homography(batch_size, 3, std_val)
    return H_left.permute(0, 2, 1) @ F_rect @ H_right


# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.bfloat16: (7.8e-3, 7.8e-3),
    torch.float16: (9.7e-4, 9.7e-4),
    torch.float32: (1e-4, 1e-5),  # TODO: Update to ~1.2e-7
    # TODO: Update to ~2.3e-16 for fp64
    torch.float64: (1e-5, 1e-5),  # TODO: BaseTester used (1.3e-6, 1e-5), but it fails for general cases
}


class BaseTester(ABC):
    @abstractmethod
    def test_smoke(self, device: Device, dtype: Dtype) -> None:
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_exception(self, device: Device, dtype: Dtype) -> None:
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_cardinality(self, device: Device, dtype: Dtype) -> None:
        raise NotImplementedError("Implement a stupid routine.")

    # TODO: add @abstractmethod
    def test_dynamo(self, device: Device, dtype: Dtype, torch_optimizer: Callable[..., Any]) -> None:
        pass  # TODO: raise NotImplementedError -- now we see a bunch of dynamo tests running by inheritance

    @abstractmethod
    def test_gradcheck(self, device: Device) -> None:
        raise NotImplementedError("Implement a stupid routine.")

    def test_module(self, device: Device, dtype: Dtype) -> None:
        pass

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
        inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        *,
        raise_exception: bool = True,
        fast_mode: bool = True,
        **kwargs: Any,
    ) -> bool:
        return gradcheck(func, inputs, raise_exception=raise_exception, fast_mode=fast_mode, **kwargs)


def generate_two_view_random_scene(
    device: Device = torch.device("cpu"), dtype: Dtype = torch.float32
) -> Dict[str, Tensor]:
    from kornia.geometry import epipolar as epi

    num_views: int = 2
    num_points: int = 30

    scene: Dict[str, Tensor] = epi.generate_scene(num_views, num_points)

    # internal parameters (same K)
    K1 = scene["K"].to(device, dtype)
    K2 = K1.clone()

    # rotation
    R1 = scene["R"][0:1].to(device, dtype)
    R2 = scene["R"][1:2].to(device, dtype)

    # translation
    t1 = scene["t"][0:1].to(device, dtype)
    t2 = scene["t"][1:2].to(device, dtype)

    # projection matrix, P = K(R|t)
    P1 = scene["P"][0:1].to(device, dtype)
    P2 = scene["P"][1:2].to(device, dtype)

    # fundamental matrix
    F_mat = epi.fundamental_from_projections(P1[..., :3, :], P2[..., :3, :])

    F_mat = epi.normalize_transformation(F_mat)

    # points 3d
    X = scene["points3d"].to(device, dtype)

    # projected points
    x1 = scene["points2d"][0:1].to(device, dtype)
    x2 = scene["points2d"][1:2].to(device, dtype)

    return {
        "K1": K1,
        "K2": K2,
        "R1": R1,
        "R2": R2,
        "t1": t1,
        "t2": t2,
        "P1": P1,
        "P2": P2,
        "F": F_mat,
        "X": X,
        "x1": x1,
        "x2": x2,
    }


def cartesian_product_of_parameters(**possible_parameters: Sequence[Any]) -> Iterator[Dict[str, Any]]:
    """Create cartesian product of given parameters."""
    parameter_names = possible_parameters.keys()
    possible_values = [possible_parameters[parameter_name] for parameter_name in parameter_names]

    for param_combination in product(*possible_values):
        yield dict(zip(parameter_names, param_combination))


def default_with_one_parameter_changed(*, default: Dict[str, Any] = {}, **possible_parameters: Any) -> Any:
    if not isinstance(default, dict):
        raise AssertionError(f"default should be a dict not a {type(default)}")

    for parameter_name, possible_values in possible_parameters.items():
        for v in possible_values:
            param_set = deepcopy(default)
            param_set[parameter_name] = v
            yield param_set


def _get_precision(device: torch.device, dtype: Dtype) -> float:
    if "xla" in device.type:
        return 1e-2
    if dtype == torch.float16:
        return 1e-3
    return 1e-4


def _get_precision_by_name(
    device: torch.device, device_target: str, tol_val: float, tol_val_default: float = 1e-4
) -> float:
    if device_target not in ["cpu", "cuda", "xla", "mps"]:
        raise ValueError(f"Invalid device name: {device_target}.")

    if device_target in device.type:
        return tol_val

    return tol_val_default


def _default_tolerances(*inputs: Any) -> Tuple[float, float]:
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
