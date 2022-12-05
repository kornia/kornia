import torch
import torch.nn.functional as TF

from kornia.core import Tensor
from kornia.testing import KORNIA_CHECK_SHAPE
from kornia.utils.misc import reduce_first_dims


def rgb_to_xyz(rgb: Tensor) -> Tensor:
    r"""Convert a RGB image to XYZ.

    .. image:: _static/img/rgb_to_xyz.png

    Args:
        rgb: RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
         XYZ version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> rgb = torch.rand(2, 3, 4, 5)
        >>> xyz = rgb_to_xyz(rgb)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(rgb, ["*", "3", "H", "W"])
    rgb, shape = reduce_first_dims(rgb, keep_last_dims=3)

    weights = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]).type_as(rgb).view(3, 3, 1, 1)

    xyz = TF.conv2d(rgb, weights, bias=None)
    return xyz.view(*shape[:-3], 3, shape[-2], shape[-1])


def xyz_to_rgb(xyz: Tensor) -> Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        xyz: XYZ Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> xyz = torch.rand(2, 3, 4, 5)
        >>> rgb = xyz_to_rgb(xyz)  # 2x3x4x5
    """
    KORNIA_CHECK_SHAPE(xyz, ["*", "3", "H", "W"])
    xyz, shape = reduce_first_dims(xyz, keep_last_dims=3)

    weights = torch.tensor([
        [3.24048134320052660, -1.5371515162713185, -0.4985363261688878],
        [-0.9692549499965682, 1.87599000148989070, 0.04155592655829280],
        [0.05564663913517720, -0.2040413383665112, 1.05731106964534430],
    ]).type_as(xyz).view(3, 3, 1, 1)

    rgb = TF.conv2d(xyz, weights, bias=None)
    return rgb.view(*shape[:-3], 3, shape[-2], shape[-1])
