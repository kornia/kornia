from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor, tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["PerspectiveGenerator"]


class PerspectiveGenerator(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: the degree of distortion, ranged from 0 to 1.
        sampling_method: ``'basic'`` | ``'area_preserving'``. Default: ``'basic'``
            If ``'basic'``, samples by translating the image corners randomly inwards.
            If ``'area_preserving'``, samples by randomly translating the image corners in any direction.
            Preserves area on average. See https://arxiv.org/abs/2104.03308 for further details.

    Returns:
        A dict of parameters to be passed for transformation.
            - start_points (Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
            - end_points (Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, distortion_scale: Union[Tensor, float] = 0.5, sampling_method: str = "basic") -> None:
        super().__init__()
        if sampling_method not in ("basic", "area_preserving"):
            raise NotImplementedError(f"Sampling method {sampling_method} not yet implemented.")
        self.distortion_scale = distortion_scale
        self.sampling_method = sampling_method

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}.")
        self.rand_val_sampler = Uniform(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.distortion_scale])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        start_points: Tensor = tensor(
            [[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]], device=_device, dtype=_dtype
        ).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = self._distortion_scale * width / 2
        fy = self._distortion_scale * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2).to(device=_device, dtype=_dtype)

        # TODO: This line somehow breaks the gradcheck
        rand_val: Tensor = _adapted_rsampling(start_points.shape, self.rand_val_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        if self.sampling_method == "basic":
            pts_norm = tensor([[[1, 1], [-1, 1], [-1, -1], [1, -1]]], device=_device, dtype=_dtype)
            offset = factor * rand_val * pts_norm
        elif self.sampling_method == "area_preserving":
            offset = 2 * factor * (rand_val - 0.5)

        end_points = start_points + offset

        return {"start_points": start_points, "end_points": end_points}
