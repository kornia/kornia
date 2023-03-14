from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import RainGenerator
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomRain(IntensityAugmentationBase2D):
    r"""Add Random Rain to the image.

     Args:
         p: probability of applying the transformation.
         number_of_drops: number of drops per image
         drop_height: height of the drop in image(same for each drops in one image)
         drop_width: width of the drop in image(same for each drops in one image)
     Shape:
         - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
         - Output: :math:`(B, C, H, W)`

     Examples:
     >>> import torch
     >>> rng = torch.manual_seed(0)
     >>> input = torch.rand(1, 1, 5, 5)
     >>> rain = RandomRain(p=1,drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1))
     >>> rain(input)
    tensor([[[[0.7843, 0.4164, 0.6583, 0.7164, 0.8444],
          [0.2766, 0.3985, 0.8437, 0.4464, 0.0751],
          [0.7116, 0.5010, 0.3791, 0.1553, 0.8987],
          [0.4833, 0.6489, 0.5527, 0.4286, 0.5395],
          [0.6714, 0.5087, 0.6090, 0.4868, 0.3147]]]])
    """

    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        number_of_drops: Tuple[int, int] = (1000, 2000),
        drop_height: Tuple[int, int] = (5, 20),
        drop_width: Tuple[int, int] = (-5, 5),
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = RainGenerator(number_of_drops, drop_height, drop_width)

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK(image.shape[1] == 3, "Number of color channels should be 3.")
        KORNIA_CHECK(len(image.shape) in (3, 4), "Wrong input dimension.")

        if len(image.shape) == 3:
            image = image[None]
        for i in range(image.shape[0]):
            number_of_drops = int(params['number_of_drops_factor'][i])
            height_of_drop = params['drop_height_factor'][i]
            width_of_drop = params['drop_width_factor'][i]

            KORNIA_CHECK(height_of_drop >= image[i].shape[1], "Height of drop should be less than image height")

            KORNIA_CHECK(width_of_drop >= image[i].shape[2], "Width of drop should be less than image width")
            KORNIA_CHECK(height_of_drop < 0, "Height should be bigger than 0")
            # Generate start coordinates for each drop

            random_y_coords = torch.randint(low=0, high=image[i].shape[1] - height_of_drop, size=[1, number_of_drops])
            if width_of_drop < 0:
                random_x_coords = torch.randint(
                    low=-width_of_drop - 1, high=image[i].shape[2], size=[1, number_of_drops]
                )
            else:
                random_x_coords = torch.randint(
                    low=0, high=image[i].shape[2] - width_of_drop, size=[1, number_of_drops]
                )
            coords = torch.concat([random_y_coords, random_x_coords], dim=0).to(image.device)

            # Generate how our drop will look like into the image
            size_of_line = max(height_of_drop, abs(width_of_drop))
            x = torch.linspace(start=0, end=height_of_drop, steps=size_of_line, dtype=torch.int32).to(image.device)
            y = torch.linspace(start=0, end=width_of_drop, steps=size_of_line, dtype=torch.int32).to(image.device)

            # Draw lines
            for k in range(x.shape[0]):
                print(coords[0] + x[k], coords[1] + y[k], image[i].shape)
                image[i, :, coords[0] + x[k], coords[1] + y[k]] = 200 / 255
        return image
