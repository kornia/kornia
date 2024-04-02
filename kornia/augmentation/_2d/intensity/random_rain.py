from __future__ import annotations

from typing import Any, Optional

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
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> rain = RandomRain(p=1,drop_height=(1,2),drop_width=(1,2),number_of_drops=(1,1))
        >>> rain(input)
        tensor([[[[0.4963, 0.7843, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])
    """

    def __init__(
        self,
        number_of_drops: tuple[int, int] = (1000, 2000),
        drop_height: tuple[int, int] = (5, 20),
        drop_width: tuple[int, int] = (-5, 5),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = RainGenerator(number_of_drops, drop_height, drop_width)

    def apply_transform(
        self, image: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Check array and drops size
        KORNIA_CHECK(image.shape[1] in {3, 1}, "Number of color channels should be 1 or 3.")
        KORNIA_CHECK(
            bool(
                torch.all(params["drop_height_factor"] <= image.shape[2])
                and torch.all(params["drop_height_factor"] > 0)
            ),
            "Height of drop should be greater than zero and less than image height.",
        )

        KORNIA_CHECK(
            bool(torch.all(torch.abs(params["drop_width_factor"]) <= image.shape[3])),
            "Width of drop should be less than image width.",
        )
        modeified_img = image.clone()
        for i in range(image.shape[0]):
            number_of_drops: int = int(params["number_of_drops_factor"][i])
            # We generate tensor with maximum number of drops, and then remove unnecessary drops.

            coordinates_of_drops: Tensor = params["coordinates_factor"][i][:number_of_drops]
            height_of_drop: int = int(params["drop_height_factor"][i])
            width_of_drop: int = int(params["drop_width_factor"][i])

            # Generate start coordinates for each drop
            random_y_coords = coordinates_of_drops[:, 0] * (image.shape[2] - height_of_drop - 1)
            if width_of_drop > 0:
                random_x_coords = coordinates_of_drops[:, 1] * (image.shape[3] - width_of_drop - 1)
            else:
                random_x_coords = coordinates_of_drops[:, 1] * (image.shape[3] + width_of_drop - 1) - width_of_drop

            coords = torch.cat([random_y_coords[None], random_x_coords[None]], dim=0).to(image.device, dtype=torch.long)

            # Generate how our drop will look like into the image
            size_of_line: int = max(height_of_drop, abs(width_of_drop))
            x = torch.linspace(start=0, end=height_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)
            y = torch.linspace(start=0, end=width_of_drop, steps=size_of_line, dtype=torch.long).to(image.device)
            # Draw lines
            for k in range(x.shape[0]):
                modeified_img[i, :, coords[0] + x[k], coords[1] + y[k]] = 200 / 255
        return modeified_img
