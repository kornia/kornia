"""
The testing package contains testing-specific utilities.
"""
from abc import ABC, abstractmethod
import importlib
from itertools import product
from copy import deepcopy

import torch
import numpy as np


__all__ = [
    'tensor_to_gradcheck_var', 'create_eye_batch', 'xla_is_available'
]


def xla_is_available() -> bool:
    """Return whether `torch_xla` is available in the system."""
    if importlib.util.find_spec("torch_xla") is not None:
        return True
    return False


def create_checkerboard(h, w, nw):
    """Creates a synthetic checkerd board of shape HxW and window size `nw`.
    """
    return np.kron([[1, 0] * nw, [0, 1] * nw] * nw,
                   np.ones((h // (2 * nw), w // (2 * nw)))).astype(np.float32)


# TODO: Isn't this function duplicated with eye_like?
def create_eye_batch(batch_size, eye_size):
    """Creates a batch of identity matrices of shape Bx3x3
    """
    return torch.eye(eye_size).view(
        1, eye_size, eye_size).expand(batch_size, -1, -1)


def create_random_homography(batch_size, eye_size, std_val=1e-3):
    """Creates a batch of random homographies of shape Bx3x3
    """
    std = torch.FloatTensor(batch_size, eye_size, eye_size)
    eye = create_eye_batch(batch_size, eye_size)
    return eye + std.uniform_(-std_val, std_val)


def tensor_to_gradcheck_var(tensor, dtype=torch.float64, requires_grad=True):
    """Converts the input tensor to a valid variable to check the gradient.
      `gradcheck` needs 64-bit floating point and requires gradient.
    """
    assert torch.is_tensor(tensor), type(tensor)
    return tensor.requires_grad_(requires_grad).type(dtype)


def compute_patch_error(x, y, h, w):
    """Compute the absolute error between patches.
    """
    return torch.abs(x - y)[..., h // 4:-h // 4, w // 4:-w // 4].mean()


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def create_rectified_fundamental_matrix(batch_size):
    """Creates a batch of rectified fundamental matrices of shape Bx3x3
    """
    F_rect = torch.tensor([[0., 0., 0.],
                           [0., 0., -1.],
                           [0., 1., 0.]]).view(1, 3, 3)
    F_repeat = F_rect.repeat(batch_size, 1, 1)
    return F_repeat


def create_random_fundamental_matrix(batch_size, std_val=1e-3):
    """Creates a batch of random fundamental matrices of shape Bx3x3
    """
    F_rect = create_rectified_fundamental_matrix(batch_size)
    H_left = create_random_homography(batch_size, 3, std_val)
    H_right = create_random_homography(batch_size, 3, std_val)
    return H_left.permute(0, 2, 1) @ F_rect @ H_right


class BaseTester(ABC):
    @abstractmethod
    def test_smoke(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_exception(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_cardinality(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_jit(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_gradcheck(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_module(self):
        raise NotImplementedError("Implement a stupid routine.")


def cartesian_product_of_parameters(**possible_parameters):
    """Creates cartesian product of given parameters
    """
    parameter_names = possible_parameters.keys()
    possible_values = [possible_parameters[parameter_name] for parameter_name in parameter_names]

    for param_combination in product(*possible_values):
        yield dict(zip(parameter_names, param_combination))


def default_with_one_parameter_changed(*, default={}, **possible_parameters):
    assert isinstance(default, dict), f"default should be a dict not a {type(default)}"

    for parameter_name, possible_values in possible_parameters.items():
        for v in possible_values:
            param_set = deepcopy(default)
            param_set[parameter_name] = v
            yield param_set


def _get_precision(device: torch.device, dtype: torch.dtype) -> float:
    if 'xla' in device.type:
        return 1e-2
    elif dtype == torch.float16:
        return 1e-3
    return 1e-4


def _get_precision_by_name(device: torch.device, device_target: str,
                           tol_val: float, tol_val_default: float = 1e-4) -> float:
    if device_target not in ['cpu', 'cuda', 'xla']:
        raise ValueError(f"Invalid device name: {device_target}.")

    if device_target in device.type:
        return tol_val

    return tol_val_default
