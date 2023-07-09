from typing import Dict, Optional, Tuple

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor
from kornia.geometry.bbox import bbox_generator
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["MosaicGenerator"]


class MosaicGenerator(RandomGeneratorBase):
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        output_size: the output tensor width and height after mosaicing.
        mosaic_grid: the number of images and image arrangement. e.g. (2, 2) means
            each output will mix 4 images in a 2x2 grid.
        start_ratio_range: top-left (x, y) position for cropping the mosaic images.

    Returns:
        A dict of parameters to be passed for transformation.
            - mosaic_ids (Tensor): a shape of (B, N) tensor, where n is the number of mosaic images.
            - src (Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (Tensor): output bounding boxes with a shape (B, 4, 2).
            - batch_shapes (Tensor): image shapes in the batch with a shape of (B, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        mosaic_grid: Tuple[int, int] = (2, 2),
        start_ratio_range: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.mosaic_grid = mosaic_grid
        self.start_ratio_range = start_ratio_range

    def __repr__(self) -> str:
        repr = (
            f"output_size={self.output_size}, mosaic_grid={self.mosaic_grid}, "
            f"start_ratio_range={self.start_ratio_range}"
        )
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.start_ratio_range_sampler = Uniform(
            torch.tensor(self.start_ratio_range[0], device=device, dtype=dtype),
            torch.tensor(self.start_ratio_range[1], device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        input_sizes = (batch_shape[-2], batch_shape[-1])
        # output_size = input_sizes if self.output_size is None else self.output_size

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.mosaic_grid])

        perm_times = self.mosaic_grid[0] * self.mosaic_grid[1]
        # Generate mosiac order in one shot
        rand_ids = torch.randperm(batch_size * (perm_times - 1), device=_device) % batch_size
        mosiac_ids = (
            torch.cat([torch.arange(0, batch_size, device=_device), rand_ids])
            .reshape(perm_times, batch_size)
            .permute(1, 0)
        )

        start_corner_factor = _adapted_rsampling(
            (batch_size, 2), self.start_ratio_range_sampler, same_on_batch=False
        ).to(device=_device, dtype=_dtype)
        start_corner_x = start_corner_factor[:, 0] * batch_shape[-2]
        start_corner_y = start_corner_factor[:, 1] * batch_shape[-1]
        crop_src = bbox_generator(
            start_corner_x,
            start_corner_y,
            start_corner_x.clone().fill_(input_sizes[0]),
            start_corner_y.clone().fill_(input_sizes[1]),
        )
        crop_dst = torch.tensor(
            [[[0, 0], [input_sizes[1] - 1, 0], [input_sizes[1] - 1, input_sizes[0] - 1], [0, input_sizes[0] - 1]]],
            device=_device,
            dtype=_dtype,
        ).repeat(batch_size, 1, 1)
        # NOTE: In case we support a list of tensor images later. For a better consistency.
        # B x 3

        batch_shapes: Tensor
        if batch_size == 0:
            batch_shapes = torch.zeros([0, 3], device=_device, dtype=torch.long)
        else:
            batch_shapes = torch.stack([torch.as_tensor(batch_shape[1:], device=_device) for _ in range(batch_size)])
        return {
            "permutation": mosiac_ids.to(device=_device, dtype=torch.long),
            "src": crop_src,
            "dst": crop_dst,
            "batch_shapes": batch_shapes,
        }
