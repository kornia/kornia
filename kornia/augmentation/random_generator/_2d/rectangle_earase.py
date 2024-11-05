from typing import Dict, Tuple, Union

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check
from kornia.core import Tensor, as_tensor, tensor, where
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["RectangleEraseGenerator"]


class RectangleEraseGenerator(RandomGeneratorBase):
    r"""Get parameters for ```erasing``` transformation for erasing transform.

    Args:
        scale (Tensor): range of size of the origin size cropped. Shape (2).
        ratio (Tensor): range of aspect ratio of the origin aspect ratio cropped. Shape (2).
        value (float): value to be filled in the erased area.

    Returns:
        A dict of parameters to be passed for transformation.
            - widths (Tensor): element-wise erasing widths with a shape of (B,).
            - heights (Tensor): element-wise erasing heights with a shape of (B,).
            - xs (Tensor): element-wise erasing x coordinates with a shape of (B,).
            - ys (Tensor): element-wise erasing y coordinates with a shape of (B,).
            - values (Tensor): element-wise filling values with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __repr__(self) -> str:
        repr = f"scale={self.scale}, resize_to={self.ratio}, value={self.value}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = as_tensor(self.scale, device=device, dtype=dtype)
        ratio = as_tensor(self.ratio, device=device, dtype=dtype)

        if not (isinstance(self.value, (int, float)) and self.value >= 0 and self.value <= 1):
            raise AssertionError(f"'value' must be a number between 0 - 1. Got {self.value}.")
        _joint_range_check(scale, "scale", bounds=(0, float("inf")))
        _joint_range_check(ratio, "ratio", bounds=(0, float("inf")))

        self.scale_sampler = UniformDistribution(scale[0], scale[1], validate_args=False)

        if ratio[0] < 1.0 and ratio[1] > 1.0:
            self.ratio_sampler1 = UniformDistribution(ratio[0], 1, validate_args=False)
            self.ratio_sampler2 = UniformDistribution(1, ratio[1], validate_args=False)
            self.index_sampler = UniformDistribution(
                tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
            )
        else:
            self.ratio_sampler = UniformDistribution(ratio[0], ratio[1], validate_args=False)
        self.uniform_sampler = UniformDistribution(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.ratio, self.scale])
        images_area = height * width
        target_areas = (
            _adapted_rsampling((batch_size,), self.scale_sampler, same_on_batch).to(device=_device, dtype=_dtype)
            * images_area
        )

        if self.ratio[0] < 1.0 and self.ratio[1] > 1.0:
            aspect_ratios1 = _adapted_rsampling((batch_size,), self.ratio_sampler1, same_on_batch)
            aspect_ratios2 = _adapted_rsampling((batch_size,), self.ratio_sampler2, same_on_batch)
            if same_on_batch:
                rand_idxs = (
                    torch.round(_adapted_rsampling((1,), self.index_sampler, same_on_batch)).repeat(batch_size).bool()
                )
            else:
                rand_idxs = torch.round(_adapted_rsampling((batch_size,), self.index_sampler, same_on_batch)).bool()
            aspect_ratios = where(rand_idxs, aspect_ratios1, aspect_ratios2)
        else:
            aspect_ratios = _adapted_rsampling((batch_size,), self.ratio_sampler, same_on_batch)

        aspect_ratios = aspect_ratios.to(device=_device, dtype=_dtype)

        # based on target areas and aspect ratios, rectangle params are computed
        heights = torch.min(
            torch.max(
                torch.round((target_areas * aspect_ratios) ** (1 / 2)), tensor(1.0, device=_device, dtype=_dtype)
            ),
            tensor(height, device=_device, dtype=_dtype),
        )

        widths = torch.min(
            torch.max(
                torch.round((target_areas / aspect_ratios) ** (1 / 2)), tensor(1.0, device=_device, dtype=_dtype)
            ),
            tensor(width, device=_device, dtype=_dtype),
        )

        xs_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        ys_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        xs = xs_ratio * (width - widths + 1)
        ys = ys_ratio * (height - heights + 1)

        return {
            "widths": widths.floor(),
            "heights": heights.floor(),
            "xs": xs.floor(),
            "ys": ys.floor(),
            "values": tensor([self.value] * batch_size, device=_device, dtype=_dtype),
        }
