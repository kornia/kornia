from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor
from kornia.utils import _extract_device_dtype


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

    def draw_line(self, img: torch.Tensor, intensity: float = 0.3) -> torch.Tensor:
        """Draws the drops on the image by changing its pixels' intensity."""
        image = img.clone()
        if len(image.shape) != 4:
            raise ValueError("Input image tensor should have shape (B, C, H, W)")
        B, C, H, W = image.shape
        # generate rain parameters
        rain_params = self.forward((B,), same_on_batch=False)
        num_drops = rain_params['number_of_drops_factor'].item()
        drop_heights = rain_params['drop_height_factor'].item()
        drop_widths = rain_params['drop_width_factor'].item()
        coordinates = rain_params['coordinates_factor']
        for b in range(B):
            for drop in range(int(num_drops)):
                x_start = int(coordinates[b, drop, 0].item() * W)
                y_start = int(coordinates[b, drop, 1].item() * H)
                x_end = x_start + drop_widths
                y_end = y_start + drop_heights
                # ensure the raindrop is within the image boundaries
                x_end = int(min(x_end, W))
                y_end = int(min(y_end, H))
                image[b, :, y_start:y_end, x_start:x_end] -= intensity  # darken the image where the raindrop is
        image = torch.clamp(image, 0, 1)  # clip values to ensure they're within [0, 1]
        return image
