from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.utils.helpers import _extract_device_dtype

from ...utils import _adapted_rsampling, _joint_range_check
from .crop import CropGenerator


class ResizedCropGenerator(CropGenerator):
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        output_size (Tuple[int, int]): expected output size of each edge.
        scale (torch.Tensor): range of size of the origin size cropped with (2,) shape.
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped with (2,) shape.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - size (torch.Tensor): element-wise cropping sizes with a shape of (B, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> random_crop_size_generator(3, (30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))
        {'size': tensor([[29., 29.],
                [27., 28.],
                [26., 29.]])}
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        scale: Union[torch.Tensor, Tuple[float, float]],
        ratio: Union[torch.Tensor, Tuple[float, float]],
    ) -> None:
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size
        if not (
            len(output_size) == 2
            and isinstance(output_size[0], (int,))
            and isinstance(output_size[1], (int,))
            and output_size[0] > 0
            and output_size[1] > 0
        ):
            raise AssertionError(f"`output_size` must be a tuple of 2 positive integers. Got {output_size}.")
        super().__init__(size=output_size, resize_to=self.output_size)  # fake an intermedia crop size

    def __repr__(self) -> str:
        repr = f"scale={self.scale}, resize_to={self.ratio}, output_size={self.output_size}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
        ratio = torch.as_tensor(self.ratio, device=device, dtype=dtype)
        _joint_range_check(scale, "scale")
        _joint_range_check(ratio, "ratio")
        self.rand_sampler = Uniform(
            torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)
        )
        self.log_ratio_sampler = Uniform(torch.log(ratio[0]), torch.log(ratio[1]), validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        size = (batch_shape[-2], batch_shape[-1])
        _device, _dtype = _extract_device_dtype([self.scale, self.ratio])

        if batch_size == 0:
            return dict(
                src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                dst=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                size=torch.zeros([0, 2], device=_device, dtype=_dtype),
            )

        rand = _adapted_rsampling((batch_size, 10), self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype)
        area = (rand * (self.scale[1] - self.scale[0]) + self.scale[0]) * size[0] * size[1]
        log_ratio = _adapted_rsampling((batch_size, 10), self.log_ratio_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        aspect_ratio = torch.exp(log_ratio)

        w = torch.sqrt(area * aspect_ratio).round().floor()
        h = torch.sqrt(area / aspect_ratio).round().floor()
        # Element-wise w, h condition
        cond = ((0 < w) * (w < size[0]) * (0 < h) * (h < size[1])).int()

        # torch.argmax is not reproducible across devices: https://github.com/pytorch/pytorch/issues/17738
        # Here, we will select the first occurrence of the duplicated elements.
        cond_bool, argmax_dim1 = ((cond.cumsum(1) == 1) & cond.bool()).max(1)
        h_out = w[torch.arange(0, batch_size, device=_device, dtype=torch.long), argmax_dim1]
        w_out = h[torch.arange(0, batch_size, device=_device, dtype=torch.long), argmax_dim1]

        if not cond_bool.all():
            # Fallback to center crop
            in_ratio = float(size[0]) / float(size[1])
            _min = self.ratio.min() if isinstance(self.ratio, torch.Tensor) else min(self.ratio)
            if in_ratio < _min:  # type:ignore
                h_ct = torch.tensor(size[0], device=_device, dtype=_dtype)
                w_ct = torch.round(h_ct / _min)
            elif in_ratio > _min:  # type:ignore
                w_ct = torch.tensor(size[1], device=_device, dtype=_dtype)
                h_ct = torch.round(w_ct * _min)
            else:  # whole image
                h_ct = torch.tensor(size[0], device=_device, dtype=_dtype)
                w_ct = torch.tensor(size[1], device=_device, dtype=_dtype)
            h_ct = h_ct.floor()
            w_ct = w_ct.floor()

            h_out = h_out.where(cond_bool, h_ct)
            w_out = w_out.where(cond_bool, w_ct)

        # Update the crop size.
        self.size = torch.stack([h_out, w_out], dim=1)
        return super().forward(batch_shape, same_on_batch)
