"""
The testing package contains testing-specific utilities.
"""
from abc import ABC, abstractmethod

import torch
import numpy as np


__all__ = [
    'tensor_to_gradcheck_var', 'create_eye_batch',
]


def create_checkerboard(h, w, nw):
    """Creates a synthetic checkerd board of shape HxW and window size `nw`.
    """
    return np.kron([[1, 0] * nw, [0, 1] * nw] * nw,
                   np.ones((h // (2 * nw), w // (2 * nw)))).astype(np.float32)


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
    def test_batch(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_jit(self):
        raise NotImplementedError("Implement a stupid routine.")

    @abstractmethod
    def test_gradcheck(self):
        raise NotImplementedError("Implement a stupid routine.")
