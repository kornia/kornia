# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)
import sys

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later")

try:
    from .version import __version__  # nopa: 401
except ImportError:
    pass

import torch

# NOTE: kornia.geomtry must go first since it is the core of the library
# and by changing the import order you might get into a circular dependencies issue.
from kornia import (
    augmentation,
    color,
    contrib,
    enhance,
    feature,
    filters,
    geometry,
    losses,
    metrics,
    morphology,
    utils,
    x,
)
from kornia.constants import pi
from kornia.testing import xla_is_available
from kornia.utils import create_meshgrid, image_to_tensor, tensor_to_image

# NOTE: we are going to expose to top level very few things



def eye_like(n: int, input: torch.Tensor) -> torch.Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def vec_like(n, tensor):
    r"""Return a 2-D tensor with a vector containing zeros with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        tensor: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.

    Returns:
        The vector with the same batch size as the input :math:`(B, N, 1)`.

    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(tensor.shape) < 1:
        raise AssertionError(tensor.shape)

    vec = torch.zeros(n, 1, device=tensor.device, dtype=tensor.dtype)
    return vec[None].repeat(tensor.shape[0], 1, 1)
