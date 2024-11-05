from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check
from kornia.core import Device, Tensor, tensor, where, zeros
from kornia.geometry.bbox import bbox_generator
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["CropGenerator", "ResizedCropGenerator", "center_crop_generator"]


class CropGenerator(RandomGeneratorBase):
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.

    Returns:
        params Dict[str, Tensor]: parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (Tensor): output bounding boxes with a shape (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, size: Union[Tuple[int, int], Tensor], resize_to: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.size = size
        self.resize_to = resize_to

    def __repr__(self) -> str:
        repr = f"crop_size={self.size}"
        if self.resize_to is not None:
            repr += f", resize_to={self.resize_to}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.rand_sampler = Uniform(tensor(0.0, device=device, dtype=dtype), tensor(1.0, device=device, dtype=dtype))

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.size if isinstance(self.size, Tensor) else None])

        if batch_size == 0:
            return {
                "src": zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": zeros([0, 4, 2], device=_device, dtype=_dtype),
            }

        input_size = (batch_shape[-2], batch_shape[-1])
        if not isinstance(self.size, Tensor):
            size = tensor(self.size, device=_device, dtype=_dtype).repeat(batch_size, 1)
        else:
            size = self.size.to(device=_device, dtype=_dtype)
        if size.shape != torch.Size([batch_size, 2]):
            raise AssertionError(
                "If `size` is a tensor, it must be shaped as (B, 2). "
                f"Got {size.shape} while expecting {torch.Size([batch_size, 2])}."
            )
        if not (input_size[0] > 0 and input_size[1] > 0 and (size > 0).all()):
            raise AssertionError(f"Got non-positive input size or size. {input_size}, {size}.")
        size = size.floor()

        x_diff = input_size[1] - size[:, 1] + 1
        y_diff = input_size[0] - size[:, 0] + 1

        # Start point will be 0 if diff < 0
        x_diff = x_diff.clamp(0)
        y_diff = y_diff.clamp(0)

        if same_on_batch:
            # If same_on_batch, select the first then repeat.
            x_start = (
                _adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(x_diff) * x_diff[0]
            ).floor()
            y_start = (
                _adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(y_diff) * y_diff[0]
            ).floor()
        else:
            x_start = (_adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(x_diff) * x_diff).floor()
            y_start = (_adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(y_diff) * y_diff).floor()
        crop_src = bbox_generator(
            x_start.view(-1).to(device=_device, dtype=_dtype),
            y_start.view(-1).to(device=_device, dtype=_dtype),
            where(size[:, 1] == 0, tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
            where(size[:, 0] == 0, tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]),
        )

        if self.resize_to is None:
            crop_dst = bbox_generator(
                tensor([0] * batch_size, device=_device, dtype=_dtype),
                tensor([0] * batch_size, device=_device, dtype=_dtype),
                size[:, 1],
                size[:, 0],
            )
            _output_size = size.to(dtype=torch.long)
        else:
            if not (
                len(self.resize_to) == 2
                and isinstance(self.resize_to[0], (int,))
                and isinstance(self.resize_to[1], (int,))
                and self.resize_to[0] > 0
                and self.resize_to[1] > 0
            ):
                raise AssertionError(f"`resize_to` must be a tuple of 2 positive integers. Got {self.resize_to}.")
            crop_dst = tensor(
                [
                    [
                        [0, 0],
                        [self.resize_to[1] - 1, 0],
                        [self.resize_to[1] - 1, self.resize_to[0] - 1],
                        [0, self.resize_to[0] - 1],
                    ]
                ],
                device=_device,
                dtype=_dtype,
            ).repeat(batch_size, 1, 1)
            _output_size = tensor(self.resize_to, device=_device, dtype=torch.long).expand(batch_size, -1)

        _input_size = tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

        return {"src": crop_src, "dst": crop_dst, "input_size": _input_size, "output_size": _output_size}


class ResizedCropGenerator(CropGenerator):
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        output_size (Tuple[int, int]): expected output size of each edge.
        scale (Tensor): range of size of the origin size cropped with (2,) shape.
        ratio (Tensor): range of aspect ratio of the origin aspect ratio cropped with (2,) shape.

    Returns:
        params Dict[str, Tensor]: parameters to be passed for transformation.
            - size (Tensor): element-wise cropping sizes with a shape of (B, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> _ = torch.manual_seed(42)
        >>> rcg = ResizedCropGenerator((30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))
        >>> out = rcg(torch.Size([1, 3, 3]))
        >>> out["src"]
        tensor([[[0., 0.],
                 [2., 0.],
                 [2., 2.],
                 [0., 2.]]])
        >>> out["dst"]
        tensor([[[ 0.,  0.],
                 [29.,  0.],
                 [29., 29.],
                 [ 0., 29.]]])
        >>> out["input_size"]
        tensor([[3, 3]])
        >>> out["output_size"]
        tensor([[30, 30]])
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        scale: Union[Tensor, Tuple[float, float]],
        ratio: Union[Tensor, Tuple[float, float]],
    ) -> None:
        if not (
            len(output_size) == 2
            and isinstance(output_size[0], (int,))
            and isinstance(output_size[1], (int,))
            and output_size[0] > 0
            and output_size[1] > 0
        ):
            raise AssertionError(f"`output_size` must be a tuple of 2 positive integers. Got {output_size}.")
        super().__init__(size=output_size, resize_to=output_size)  # fake an intermedia crop size
        self.scale = scale
        self.ratio = ratio
        self.output_size = output_size

    def __repr__(self) -> str:
        repr = f"scale={self.scale}, resize_to={self.ratio}, output_size={self.output_size}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
        ratio = torch.as_tensor(self.ratio, device=device, dtype=dtype)
        _joint_range_check(scale, "scale")
        _joint_range_check(ratio, "ratio")
        self.rand_sampler = Uniform(tensor(0.0, device=device, dtype=dtype), tensor(1.0, device=device, dtype=dtype))
        self.log_ratio_sampler = Uniform(torch.log(ratio[0]), torch.log(ratio[1]), validate_args=False)

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        size = (batch_shape[-2], batch_shape[-1])
        _device, _dtype = _extract_device_dtype([self.scale, self.ratio])

        if batch_size == 0:
            return {
                "src": zeros([0, 4, 2], device=_device, dtype=_dtype),
                "dst": zeros([0, 4, 2], device=_device, dtype=_dtype),
                "size": zeros([0, 2], device=_device, dtype=_dtype),
            }

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
            _min = float(self.ratio.min()) if isinstance(self.ratio, Tensor) else min(self.ratio)
            if in_ratio < _min:
                h_ct = tensor(size[0], device=_device, dtype=_dtype)
                w_ct = torch.round(h_ct / _min)
            elif in_ratio > _min:
                w_ct = tensor(size[1], device=_device, dtype=_dtype)
                h_ct = torch.round(w_ct * _min)
            else:  # whole image
                h_ct = tensor(size[0], device=_device, dtype=_dtype)
                w_ct = tensor(size[1], device=_device, dtype=_dtype)
            h_ct = h_ct.floor()
            w_ct = w_ct.floor()

            h_out = h_out.where(cond_bool, h_ct)
            w_out = w_out.where(cond_bool, w_ct)

        # Update the crop size.
        self.size = torch.stack([h_out, w_out], dim=1)
        return super().forward(batch_shape, same_on_batch)


def center_crop_generator(
    batch_size: int, height: int, width: int, size: Tuple[int, int], device: Device = torch.device("cpu")
) -> Dict[str, Tensor]:
    r"""Get parameters for ```center_crop``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (h, w).
        device (Device): the device on which the random numbers will be generated. Default: cpu.

    Returns:
        params Dict[str, Tensor]: parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (Tensor): output bounding boxes with a shape (B, 4, 2).

    Note:
        No random number will be generated.
    """
    _common_param_check(batch_size)
    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")
    if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
    if not (height >= size[0] and width >= size[1]):
        raise AssertionError(f"Crop size must be smaller than input size. Got ({height}, {width}) and {size}.")

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = height, width

    # compute start/end offsets
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = int(src_w_half - dst_w_half)
    start_y = int(src_h_half - dst_h_half)

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: Tensor = tensor(
        [[[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]], device=device, dtype=torch.long
    ).expand(batch_size, -1, -1)

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: Tensor = tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], device=device, dtype=torch.long
    ).expand(batch_size, -1, -1)

    _input_size = tensor((height, width), device=device, dtype=torch.long).expand(batch_size, -1)
    _output_size = tensor(size, device=device, dtype=torch.long).expand(batch_size, -1)

    return {"src": points_src, "dst": points_dst, "input_size": _input_size, "output_size": _output_size}
