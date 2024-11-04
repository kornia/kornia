from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor, as_tensor, stack, tensor
from kornia.utils.helpers import _extract_device_dtype


class PerspectiveGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: controls the degree of distortion and ranges from 0 to 1.

    Returns:
        A dict of parameters to be passed for transformation.
            - src (Tensor): perspective source bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): perspective target bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, distortion_scale: Union[Tensor, float] = 0.5) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}")
        self.rand_sampler = Uniform(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        depth = batch_shape[-3]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.distortion_scale])

        start_points: Tensor = tensor(
            [
                [
                    [0.0, 0, 0],
                    [width - 1, 0, 0],
                    [width - 1, height - 1, 0],
                    [0, height - 1, 0],
                    [0.0, 0, depth - 1],
                    [width - 1, 0, depth - 1],
                    [width - 1, height - 1, depth - 1],
                    [0, height - 1, depth - 1],
                ]
            ],
            device=_device,
            dtype=_dtype,
        ).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = self._distortion_scale * width / 2
        fy = self._distortion_scale * height / 2
        fz = self._distortion_scale * depth / 2

        factor = stack([fx, fy, fz], 0).view(-1, 1, 3).to(device=_device, dtype=_dtype)

        rand_val: Tensor = _adapted_rsampling(start_points.shape, self.rand_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        pts_norm = tensor(
            [[[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]],
            device=_device,
            dtype=_dtype,
        )
        end_points = start_points + factor * rand_val * pts_norm

        return {"start_points": start_points, "end_points": end_points}
