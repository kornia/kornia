import torch

from ..functional import apply_erase_rectangles
from ..random_generator import random_rectangles_params_generator


def _cutout(input: torch.Tensor, percentage: float = 0.2) -> torch.Tensor:
    r"""Cutout on an image tensor.

    Args:
        percentage (float): range of size of the origin size cropped.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> input = torch.randn(1, 1, 5, 5)
        >>> _cutout(input, 0.4)
        tensor([[[[ 1.2500e+02,  1.2500e+02, -2.5058e-01, -4.3388e-01,  8.4871e-01],
                  [ 1.2500e+02,  1.2500e+02, -2.1152e+00,  3.2227e-01, -1.5771e-01],
                  [ 1.4437e+00,  2.6605e-01,  1.6646e-01,  8.7438e-01, -1.4347e-01],
                  [-1.1161e-01, -6.1358e-01,  1.2590e+00,  2.0050e+00,  5.3737e-02],
                  [ 6.1806e-01, -4.1280e-01, -8.4106e-01, -2.3160e+00, -1.0231e-01]]]])
    """
    batch_size, c, h, w = input.size()
    output = input.clone()
    params = random_rectangles_params_generator(
        input.size(0), h, w, torch.tensor([percentage ** 2, percentage ** 2]),
        torch.tensor([1., 1.]), value=(125, 123, 114) if c == 3 else 125, same_on_batch=True)
    output = apply_erase_rectangles(input, params)
    return output
