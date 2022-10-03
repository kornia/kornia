"""The testing package contains testing-specific utilities."""
import contextlib
import importlib
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast

import torch
from torch import Tensor
from torch.nn import Parameter

__all__ = ['tensor_to_gradcheck_var', 'create_eye_batch', 'xla_is_available', 'assert_close']


def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


def is_mps_tensor_safe(x: Tensor) -> bool:
    """Return whether tensor is on MPS device."""
    return 'mps' in str(x.device)


# TODO: Isn't this function duplicated with eye_like?
def create_eye_batch(batch_size, eye_size, device=None, dtype=None):
    """Create a batch of identity matrices of shape Bx3x3."""
    return torch.eye(eye_size, device=device, dtype=dtype).view(1, eye_size, eye_size).expand(batch_size, -1, -1)


def create_random_homography(batch_size, eye_size, std_val=1e-3):
    """Create a batch of random homographies of shape Bx3x3."""
    std = torch.FloatTensor(batch_size, eye_size, eye_size)
    eye = create_eye_batch(batch_size, eye_size)
    return eye + std.uniform_(-std_val, std_val)


def tensor_to_gradcheck_var(tensor, dtype=torch.float64, requires_grad=True):
    """Convert the input tensor to a valid variable to check the gradient.

    `gradcheck` needs 64-bit floating point and requires gradient.
    """
    if not torch.is_tensor(tensor):
        raise AssertionError(type(tensor))
    return tensor.requires_grad_(requires_grad).type(dtype)


def dict_to(data: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out: dict = {}
    for key, val in data.items():
        out[key] = val.to(device, dtype) if isinstance(val, torch.Tensor) else val
    return out


def compute_patch_error(x, y, h, w):
    """Compute the absolute error between patches."""
    return torch.abs(x - y)[..., h // 4 : -h // 4, w // 4 : -w // 4].mean()


def check_is_tensor(obj):
    """Check whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(obj)}")


def create_rectified_fundamental_matrix(batch_size):
    """Create a batch of rectified fundamental matrices of shape Bx3x3."""
    F_rect = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).view(1, 3, 3)
    F_repeat = F_rect.repeat(batch_size, 1, 1)
    return F_repeat


def create_random_fundamental_matrix(batch_size, std_val=1e-3):
    """Create a batch of random fundamental matrices of shape Bx3x3."""
    F_rect = create_rectified_fundamental_matrix(batch_size)
    H_left = create_random_homography(batch_size, 3, std_val)
    H_right = create_random_homography(batch_size, 3, std_val)
    return H_left.permute(0, 2, 1) @ F_rect @ H_right


class BaseTester(ABC):
    DTYPE_PRECISIONS = {torch.float16: (1e-3, 1e-3), torch.float32: (1.3e-6, 1e-5), torch.float64: (1.3e-6, 1e-5)}

    @abstractmethod
    def test_smoke(self, device, dtype):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_exception(self, device, dtype):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_cardinality(self, device, dtype):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_jit(self, device, dtype):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_gradcheck(self, device):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_module(self, device, dtype):
        raise NotImplementedError("Implement a stupid routine.")

    def assert_close(
        self,
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
        if isinstance(actual, Parameter):
            actual = actual.data
        if isinstance(expected, Parameter):
            expected = expected.data

        if 'xla' in actual.device.type or 'xla' in expected.device.type:
            rtol, atol = 1e-2, 1e-2

        if rtol is None and atol is None:
            actual_rtol, actual_atol = self.DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
            expected_rtol, expected_atol = self.DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
            rtol, atol = max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)

            # halve the tolerance if `low_tolerance` is true
            rtol = math.sqrt(rtol) if low_tolerance else rtol
            atol = math.sqrt(atol) if low_tolerance else atol

        return assert_close(actual, expected, rtol=rtol, atol=atol)


def generate_two_view_random_scene(
    device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    from kornia.geometry import epipolar as epi

    num_views: int = 2
    num_points: int = 30

    scene: Dict[str, torch.Tensor] = epi.generate_scene(num_views, num_points)

    # internal parameters (same K)
    K1 = scene['K'].to(device, dtype)
    K2 = K1.clone()

    # rotation
    R1 = scene['R'][0:1].to(device, dtype)
    R2 = scene['R'][1:2].to(device, dtype)

    # translation
    t1 = scene['t'][0:1].to(device, dtype)
    t2 = scene['t'][1:2].to(device, dtype)

    # projection matrix, P = K(R|t)
    P1 = scene['P'][0:1].to(device, dtype)
    P2 = scene['P'][1:2].to(device, dtype)

    # fundamental matrix
    F_mat = epi.fundamental_from_projections(P1[..., :3, :], P2[..., :3, :])

    F_mat = epi.normalize_transformation(F_mat)

    # points 3d
    X = scene['points3d'].to(device, dtype)

    # projected points
    x1 = scene['points2d'][0:1].to(device, dtype)
    x2 = scene['points2d'][1:2].to(device, dtype)

    return dict(K1=K1, K2=K2, R1=R1, R2=R2, t1=t1, t2=t2, P1=P1, P2=P2, F=F_mat, X=X, x1=x1, x2=x2)


def cartesian_product_of_parameters(**possible_parameters):
    """Create cartesian product of given parameters."""
    parameter_names = possible_parameters.keys()
    possible_values = [possible_parameters[parameter_name] for parameter_name in parameter_names]

    for param_combination in product(*possible_values):
        yield dict(zip(parameter_names, param_combination))


def default_with_one_parameter_changed(*, default={}, **possible_parameters):
    if not isinstance(default, dict):
        raise AssertionError(f"default should be a dict not a {type(default)}")

    for parameter_name, possible_values in possible_parameters.items():
        for v in possible_values:
            param_set = deepcopy(default)
            param_set[parameter_name] = v
            yield param_set


def _get_precision(device: torch.device, dtype: torch.dtype) -> float:
    if 'xla' in device.type:
        return 1e-2
    if dtype == torch.float16:
        return 1e-3
    return 1e-4


def _get_precision_by_name(
    device: torch.device, device_target: str, tol_val: float, tol_val_default: float = 1e-4
) -> float:
    if device_target not in ['cpu', 'cuda', 'xla']:
        raise ValueError(f"Invalid device name: {device_target}.")

    if device_target in device.type:
        return tol_val

    return tol_val_default


try:
    # torch.testing.assert_close is only available for torch>=1.9
    from torch.testing import assert_close as _assert_close  # type: ignore
    from torch.testing._core import _get_default_tolerance  # type: ignore

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
    # TODO: remove this branch if kornia relies on torch>=1.9
    from torch.testing import assert_allclose as _assert_close

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
            return _assert_close(actual, expected, rtol=rtol, atol=atol, **kwargs)
        except ValueError as error:
            raise UsageError(str(error)) from error


# Logger api
def KORNIA_CHECK_SHAPE(x, shape: List[str]) -> None:
    # Desired shape here is list and not tuple, because torch.jit
    # does not like variable-length tuples
    KORNIA_CHECK_IS_TENSOR(x)
    if '*' == shape[0]:
        start_idx: int = 1
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif '*' == shape[-1]:
        start_idx: int = 0
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        start_idx = 0
        x_shape_to_check = x.shape

    for i in range(start_idx, len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            raise TypeError(f"{x} shape should be must be [{shape}]. Got {x.shape}")


def KORNIA_CHECK(condition: bool, msg: Optional[str] = None):
    if not condition:
        raise Exception(f"{condition} not true.\n{msg}")


def KORNIA_UNWRAP(maybe_obj, typ):
    return cast(typ, maybe_obj)


def KORNIA_CHECK_TYPE(x, typ, msg: Optional[str] = None):
    if not isinstance(x, typ):
        raise TypeError(f"Invalid type: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_TENSOR(x, msg: Optional[str] = None):
    if not isinstance(x, Tensor):
        raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_SAME_DEVICES(tensors: List[Tensor], msg: Optional[str] = None):
    KORNIA_CHECK(isinstance(tensors, list) and len(tensors) >= 1, "Expected a list with at least one element")
    if not all(tensors[0].device == x.device for x in tensors):
        raise Exception(f"Not same device for tensors. Got: {[x.device for x in tensors]}.\n{msg}")


def KORNIA_CHECK_IS_COLOR(x: Tensor, msg: Optional[str] = None):
    if len(x.shape) < 3 or x.shape[-3] != 3:
        raise TypeError(f"Not a color tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_GRAY(x: Tensor, msg: Optional[str] = None):
    if len(x.shape) < 2 or (len(x.shape) >= 3 and x.shape[-3] != 1):
        raise TypeError(f"Not a gray tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_COLOR_OR_GRAY(x: Tensor, msg: Optional[str] = None):
    if len(x.shape) < 3 or x.shape[-3] not in [1, 3]:
        raise TypeError(f"Not a color or gray tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_SAME_DEVICE(x: Tensor, y: Tensor):
    if x.device != y.device:
        raise TypeError(f"Not same device for tensors. Got: {x.device} and {y.device}")


def KORNIA_CHECK_DM_DESC(desc1: Tensor, desc2: Tensor, dm: Tensor):
    if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
        message = f"""distance matrix shape {dm.shape} is not
                      consistent with descriptors shape: desc1 {desc1.shape}
                      desc2 {desc2.shape}"""
        raise TypeError(message)


def KORNIA_CHECK_LAF(laf: Tensor) -> None:
    """Auxiliary function, which verifies that input.

    Args:
        laf: [BxNx2x3] shape.
    """
    KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"])
