import enum
from typing import Dict, Optional, Tuple, Union, cast

import random
import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


def shift_image(image, value):
    """
    Shift image by a certain value. Used for shifting a separate channel.
    If a pixel value is greater than maximum value, the pixel is set to a maximum value.

    Note:
        Since RandomRGBShift takes only images of [0, 1] interval, maximum value for
        a pixel after shift is 1.
    """
    max_value = torch.ones(image.shape)
    image = torch.min(max_value, image + value)
    return image


def shift_rgb(image, r_shift, g_shift, b_shift):
    """
    Shift each image's channel by either r_shift for red, g_shift for green and b_shift for blue channels.
    """
    if r_shift == g_shift == b_shift:
        return image + r_shift

    shifted = torch.empty_like(image)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        shifted[:, i, :, :] = shift_image(image[:, i, :, :], shift)

    return shifted


class RandomRGBShift(IntensityAugmentationBase2D):
    """

    Randomly shift each channel of an image.

    Args:
        r_shift_limit: maximum value up to which the shift value can be generated for red channel; 
          should be in the interval of [0, 1]
        g_shift_limit: maximum value up to which the shift value can be generated for green channel;
          should be in the interval of [0, 1]
        b_shift_limit: maximum value up to which the shift value can be generated for blue channel;
          should be in the interval of [0, 1]
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Note:
        Input tensor must be float and normalized into [0, 1].

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 3, 5, 5)
        >>> aug = RandomRGBShift(0, 0, 0)
        >>> params = aug.generate_parameters()
        >>> ((input == aug(input, params)).double()).all()
        tensor(True)

        >>> random.seed(42)
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 3, 5, 5)
        >>> input
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]],
        <BLANKLINE>
                 [[0.5529, 0.9527, 0.0362, 0.1852, 0.3734],
                  [0.3051, 0.9320, 0.1759, 0.2698, 0.1507],
                  [0.0317, 0.2081, 0.9298, 0.7231, 0.7423],
                  [0.5263, 0.2437, 0.5846, 0.0332, 0.1387],
                  [0.2422, 0.8155, 0.7932, 0.2783, 0.4820]],
        <BLANKLINE>
                 [[0.8198, 0.9971, 0.6984, 0.5675, 0.8352],
                  [0.2056, 0.5932, 0.1123, 0.1535, 0.2417],
                  [0.7262, 0.7011, 0.2038, 0.6511, 0.7745],
                  [0.4369, 0.5191, 0.6159, 0.8102, 0.9801],
                  [0.1147, 0.3168, 0.6965, 0.9143, 0.9351]]]])
        >>> aug = RandomRGBShift(p=1.)
        >>> params = aug.generate_parameters()
        >>> params
        {'r_shift': 0.13942679845788375, 'g_shift': -0.47498924477733306, 'b_shift': -0.22497068163088074}
        >>> aug(input, params)
        tensor([[[[ 0.6357,  0.9076,  0.2279,  0.2715,  0.4468],
                  [ 0.7735,  0.6295,  1.0000,  0.5951,  0.7717],
                  [ 0.4883,  0.5411,  0.1618,  0.3083,  0.4333],
                  [ 0.6579,  0.8371,  0.9394,  0.3005,  0.4217],
                  [ 0.8210,  1.0000,  0.5365,  1.0000,  0.5588]],
        <BLANKLINE>
                 [[ 0.0779,  0.4777, -0.4388, -0.2898, -0.1016],
                  [-0.1699,  0.4570, -0.2991, -0.2052, -0.3243],
                  [-0.4433, -0.2669,  0.4548,  0.2481,  0.2673],
                  [ 0.0513, -0.2313,  0.1096, -0.4418, -0.3363],
                  [-0.2328,  0.3405,  0.3182, -0.1967,  0.0070]],
        <BLANKLINE>
                 [[ 0.5948,  0.7721,  0.4735,  0.3426,  0.6103],
                  [-0.0194,  0.3682, -0.1126, -0.0715,  0.0167],
                  [ 0.5013,  0.4761, -0.0211,  0.4261,  0.5495],
                  [ 0.2119,  0.2941,  0.3909,  0.5852,  0.7551],
                  [-0.1103,  0.0918,  0.4715,  0.6893,  0.7101]]]])
    """
    # Note: Extra params, inplace=False in Torchvision.

    def __init__(
        self,
        r_shift_limit: int = 0.5,
        g_shift_limit: int = 0.5,
        b_shift_limit: int = 0.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)

        self.r_shift_limit = (-r_shift_limit, r_shift_limit)
        self.g_shift_limit = (-g_shift_limit, g_shift_limit)
        self.b_shift_limit = (-b_shift_limit, b_shift_limit)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        return shift_rgb(input, params["r_shift"], params["g_shift"], params["b_shift"])

    def generate_parameters(self) -> Dict[str, Tensor]:
        r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

        return dict(r_shift=r_shift, g_shift=g_shift, b_shift=b_shift)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
