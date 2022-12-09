import torch
import torch.nn.functional as F

from kornia.core import Tensor, tensor
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.misc import reduce_first_dims

_RGB2SEPIA_WEIGHTS = tensor(
    [[[[0.393]], [[0.769]], [[0.189]]], [[[0.349]], [[0.686]], [[0.168]]], [[[0.272]], [[0.534]], [[0.131]]]]
)  # 3x3x1x1


def sepia_from_rgb(rgb: Tensor, rescale: bool = True, eps: float = 1e-6) -> Tensor:
    r"""Apply to a tensor the sepia filter.

    Args:
        rgb: the input tensor with shape of :math:`(*, C, H, W)`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`(*, C, H, W)`.

    Example:
        >>> rgb = torch.ones(3, 1, 1)
        >>> sepia_from_rgb(rgb, rescale=False)
        tensor([[[1.3510]],
        <BLANKLINE>
                [[1.2030]],
        <BLANKLINE>
                [[0.9370]]])
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3, return_shape=True)
    sepia = F.conv2d(rgb, _RGB2SEPIA_WEIGHTS.type_as(rgb), bias=None)
    sepia = sepia.view(*shape[:-3], 3, shape[-2], shape[-1])

    if rescale:
        max_values = sepia.amax(dim=-1).amax(dim=-1)
        sepia = sepia / (max_values[..., None, None] + eps)

    return sepia
