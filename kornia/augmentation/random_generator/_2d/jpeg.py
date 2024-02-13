from typing import Dict, List, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["JPEGGenerator"]


class JPEGGenerator(RandomGeneratorBase):
    r"""Generate random JPEG augmentation parameters for a batch.

    Args:
        jpeg_quality: The RandomJPEG quality to apply

    Returns:
        A dict of parameters to be passed for transformation.
            - jpeg_quality: element-wise contrast factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        jpeg_quality: Union[Tensor, float, Tuple[float, float], List[float]] = 50.0,
    ) -> None:
        super().__init__()
        self.jpeg_quality: Union[Tensor, float, Tuple[float, float], List[float]] = jpeg_quality

    def __repr__(self) -> str:
        return f"RandomJPEG quality={self.jpeg_quality}"

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        jpeg_quality = _range_bound(
            self.jpeg_quality, "jpeg_quality", center=50.0, bounds=(1, 100), device=device, dtype=dtype
        )

        _joint_range_check(jpeg_quality, "jpeg_quality", (1, 100))

        self.jpeg_quality_sampler = UniformDistribution(jpeg_quality[0], jpeg_quality[1], validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.jpeg_quality])
        jpeg_quality_value = _adapted_rsampling((batch_size,), self.jpeg_quality_sampler, same_on_batch)
        return {"jpeg_quality": jpeg_quality_value.to(device=_device, dtype=_dtype)}
