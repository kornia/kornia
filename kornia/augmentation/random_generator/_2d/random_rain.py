from __future__ import annotations

import math
import random

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils import _extract_device_dtype, draw_line


class RainGenerator(RandomGeneratorBase):
    def __init__(
        self, number_of_drops: tuple[int, int], drop_height: tuple[int, int], drop_width: tuple[int, int]
    ) -> None:
        super().__init__()
        self.number_of_drops = number_of_drops
        self.drop_height = drop_height
        self.drop_width = drop_width

    def __repr__(self) -> str:
        repr = f"number_of_drops={self.number_of_drops}, drop_height={self.drop_height}, drop_width={self.drop_width}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        number_of_drops = _range_bound(
            self.number_of_drops,
            'number_of_drops',
            center=self.number_of_drops[0] / 2 + self.number_of_drops[1] / 2,
            bounds=(self.number_of_drops[0], self.number_of_drops[1] + 1),
        ).to(device)
        drop_height = _range_bound(
            self.drop_height,
            'drop_height',
            center=self.drop_height[0] / 2 + self.drop_height[1] / 2,
            bounds=(self.drop_height[0], self.drop_height[1] + 1),
        ).to(device)
        drop_width = _range_bound(
            self.drop_width,
            'drop_width',
            center=self.drop_width[0] / 2 + self.drop_width[1] / 2,
            bounds=(self.drop_width[0], self.drop_width[1] + 1),
        ).to(device)

        drop_coordinates = _range_bound((0, 1), 'drops_coordinate', center=0.5, bounds=(0, 1)).to(
            device=device, dtype=dtype
        )
        self.number_of_drops_sampler = UniformDistribution(number_of_drops[0], number_of_drops[1], validate_args=False)
        self.drop_height_sampler = UniformDistribution(drop_height[0], drop_height[1], validate_args=False)
        self.drop_width_sampler = UniformDistribution(drop_width[0], drop_width[1], validate_args=False)
        self.coordinates_sampler = UniformDistribution(drop_coordinates[0], drop_coordinates[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.drop_width, self.drop_height, self.number_of_drops])
        # self.ksize_factor.expand((batch_size, -1))
        number_of_drops_factor = _adapted_rsampling((batch_size,), self.number_of_drops_sampler).to(
            device=_device, dtype=torch.long
        )
        drop_height_factor = _adapted_rsampling((batch_size,), self.drop_height_sampler, same_on_batch).to(
            device=_device, dtype=torch.long
        )
        drop_width_factor = _adapted_rsampling((batch_size,), self.drop_width_sampler, same_on_batch).to(
            device=_device, dtype=torch.long
        )
        coordinates_factor = _adapted_rsampling(
            (batch_size, int(number_of_drops_factor.max().item()), 2),
            self.coordinates_sampler,
            same_on_batch=same_on_batch,
        ).to(device=_device)
        return {
            'number_of_drops_factor': number_of_drops_factor,
            'coordinates_factor': coordinates_factor,
            'drop_height_factor': drop_height_factor,
            'drop_width_factor': drop_width_factor,
        }


def random_rain_augmentation(
    image: torch.Tensor,
    num_raindrops: int = 100,
    min_length: int = 5,
    max_length: int = 15,
    color: torch.Tensor = torch.tensor([255]),
) -> torch.Tensor:
    r"""Apply random rain augmentation to the input image.

    Args:
        image: the input image with shape :math`(C,H,W)`.
        num_raindrops: number of raindrops to draw.
        min_length: minimum length of a raindrop.
        max_length: maximum length of a raindrop.
        color: the color of the raindrops.

    Return:
        The image with raindrops.
    Doctests:
    >>> import torch
    >>> image = torch.zeros(3, 10, 10) # A black 10x10 image with 3 channels
    >>> output = random_rain_augmentation(image, 0) # No raindrops
    >>> torch.equal(output, image) # Expect True since no raindrops were added
    True

    >>> output_with_rain = random_rain_augmentation(image, 1000) # Lots of raindrops
    >>> torch.equal(output_with_rain, image) # Expect False since raindrops were added
    False

    >>> # Test with custom color
    >>> custom_color = torch.tensor([50])
    >>> output_custom_color = random_rain_augmentation(image, 1000, color=custom_color)
    >>> torch.any(output_custom_color == 50) # Expect True since raindrops of color 50 were added
    True
    """
    H, W = image.shape[1], image.shape[2]

    for _ in range(num_raindrops):
        # Randomly select the starting point
        start_x = random.randint(0, W - 1)
        start_y = random.randint(0, H - 1)
        # Randomly select the length and angle of the raindrop
        length = random.uniform(min_length, max_length)
        angle = random.uniform(math.pi / 4, 3 * math.pi / 4)  # Rain typically falls at angles between 45 to 135 degrees
        # Compute end point using trigonometry
        end_x = start_x + length * math.cos(angle)
        end_y = start_y + length * math.sin(angle)
        # Clip the coordinates to be within the image boundaries
        end_x = min(max(0, end_x), W - 1)
        end_y = min(max(0, end_y), H - 1)
        # Convert the start and end points to torch tensors
        p1 = torch.tensor([start_x, start_y])
        p2 = torch.tensor([end_x, end_y])
        # Draw the raindrop
        image = draw_line(image, p1, p2, color)

    return image
