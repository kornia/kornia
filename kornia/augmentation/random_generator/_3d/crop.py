from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor, tensor, zeros
from kornia.geometry.bbox import bbox_generator3d
from kornia.utils.helpers import _extract_device_dtype


class _CropGeneratorBase3D(RandomGeneratorBase):
    has_fit_batch_prob = True

    def fit_batch_prob(
        self, batch_shape: torch.Size, batch_prob: Tensor, params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        to_apply = batch_prob.round()
        batch_size = batch_shape[0]
        size = tensor(
            (batch_shape[-3], batch_shape[-2], batch_shape[-1]), device=params["dst"].device, dtype=params["dst"].dtype
        ).repeat(batch_size, 1)
        crop_src = bbox_generator3d(
            tensor([0] * batch_size, device=params["dst"].device, dtype=params["dst"].dtype),
            tensor([0] * batch_size, device=params["dst"].device, dtype=params["dst"].dtype),
            tensor([0] * batch_size, device=params["dst"].device, dtype=params["dst"].dtype),
            size[:, 2] - 1,
            size[:, 1] - 1,
            size[:, 0] - 1,
        )
        crop_src = params["src"] * to_apply[:, None, None] + crop_src * (1 - to_apply[:, None, None])
        crop_dst = params["dst"] * to_apply[:, None, None] + crop_src * (1 - to_apply[:, None, None])
        return dict(src=crop_src, dst=crop_dst)


class CropGenerator3D(_CropGeneratorBase3D):
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        size (tuple): Desired size of the crop operation, like (d, h, w).
            If tensor, it must be (B, 3).
        resize_to (tuple): Desired output size of the crop, like (d, h, w). If None, no resize will be performed.

    Returns:
        A dict of parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    has_fit_batch_prob = True

    def __init__(
        self, size: Union[Tuple[int, int, int], Tensor], resize_to: Optional[Tuple[int, int, int]] = None
    ) -> None:
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

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size, _, depth, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.size if isinstance(self.size, Tensor) else None])

        if not isinstance(self.size, Tensor):
            size = tensor(self.size, device=_device, dtype=_dtype).repeat(batch_size, 1)
        else:
            size = self.size.to(device=_device, dtype=_dtype)
        if size.shape != torch.Size([batch_size, 3]):
            raise AssertionError(
                "If `size` is a tensor, it must be shaped as (B, 3). "
                f"Got {size.shape} while expecting {torch.Size([batch_size, 3])}."
            )
        if not (
            isinstance(depth, (int,))
            and isinstance(height, (int,))
            and isinstance(width, (int,))
            and depth > 0
            and height > 0
            and width > 0
        ):
            raise AssertionError(f"`batch_shape` should not contain negative values. Got {(batch_shape)}.")

        x_diff = width - size[:, 2] + 1
        y_diff = height - size[:, 1] + 1
        z_diff = depth - size[:, 0] + 1

        if (x_diff < 0).any() or (y_diff < 0).any() or (z_diff < 0).any():
            raise ValueError(
                f"input_size {(depth, height, width)} cannot be smaller than crop size {str(size)} in any dimension."
            )

        if batch_size == 0:
            return dict(
                src=zeros([0, 8, 3], device=_device, dtype=_dtype), dst=zeros([0, 8, 3], device=_device, dtype=_dtype)
            )

        x_start = _adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype)
        y_start = _adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype)
        z_start = _adapted_rsampling((batch_size,), self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        x_start = (x_start * x_diff).floor()
        y_start = (y_start * y_diff).floor()
        z_start = (z_start * z_diff).floor()

        crop_src = bbox_generator3d(
            x_start.view(-1), y_start.view(-1), z_start.view(-1), size[:, 2] - 1, size[:, 1] - 1, size[:, 0] - 1
        )

        if self.resize_to is None:
            crop_dst = bbox_generator3d(
                tensor([0] * batch_size, device=_device, dtype=_dtype),
                tensor([0] * batch_size, device=_device, dtype=_dtype),
                tensor([0] * batch_size, device=_device, dtype=_dtype),
                size[:, 2] - 1,
                size[:, 1] - 1,
                size[:, 0] - 1,
            )
        else:
            if not (
                len(self.resize_to) == 3
                and isinstance(self.resize_to[0], (int,))
                and isinstance(self.resize_to[1], (int,))
                and isinstance(self.resize_to[2], (int,))
                and self.resize_to[0] > 0
                and self.resize_to[1] > 0
                and self.resize_to[2] > 0
            ):
                raise AssertionError(f"`resize_to` must be a tuple of 3 positive integers. Got {self.resize_to}.")
            crop_dst = tensor(
                [
                    [
                        [0, 0, 0],
                        [self.resize_to[-1] - 1, 0, 0],
                        [self.resize_to[-1] - 1, self.resize_to[-2] - 1, 0],
                        [0, self.resize_to[-2] - 1, 0],
                        [0, 0, self.resize_to[-3] - 1],
                        [self.resize_to[-1] - 1, 0, self.resize_to[-3] - 1],
                        [self.resize_to[-1] - 1, self.resize_to[-2] - 1, self.resize_to[-3] - 1],
                        [0, self.resize_to[-2] - 1, self.resize_to[-3] - 1],
                    ]
                ],
                device=_device,
                dtype=_dtype,
            ).repeat(batch_size, 1, 1)

        return dict(src=crop_src.to(device=_device), dst=crop_dst.to(device=_device))


class CenterCropGenerator3D(_CropGeneratorBase3D):
    r"""Get parameters for ```center_crop3d``` transformation for center crop transform.

    Args:
        size: Desired size of the crop operation, like (d, h, w).

    Returns:
        params: parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        if not isinstance(size, (tuple, list)) and len(size) == 3:
            raise ValueError(f"Input size must be a tuple/list of length 3. Got {size}")
        self.size = size

    def __repr__(self) -> str:
        repr = f"size={self.size}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        pass

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        depth, height, width = (batch_shape[-3], batch_shape[-2], batch_shape[-1])
        if not (depth >= self.size[0] and height >= self.size[1] and width >= self.size[2]):
            raise AssertionError(
                f"Crop size must be smaller than input size. Got ({depth}, {height}, {width}) and {self.size}."
            )

        # unpack input sizes
        dst_d, dst_h, dst_w = self.size
        src_d, src_h, src_w = (depth, height, width)

        # compute start/end offsets
        dst_d_half = dst_d / 2
        dst_h_half = dst_h / 2
        dst_w_half = dst_w / 2
        src_d_half = src_d / 2
        src_h_half = src_h / 2
        src_w_half = src_w / 2

        start_x = src_w_half - dst_w_half
        start_y = src_h_half - dst_h_half
        start_z = src_d_half - dst_d_half

        end_x = start_x + dst_w - 1
        end_y = start_y + dst_h - 1
        end_z = start_z + dst_d - 1
        # [x, y, z] origin
        # top-left-front, top-right-front, bottom-right-front, bottom-left-front
        # top-left-back, top-right-back, bottom-right-back, bottom-left-back
        # Note: DeprecationWarning: an integer is required (got type float).
        # Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
        points_src: Tensor = tensor(
            [
                [
                    [int(start_x), int(start_y), int(start_z)],
                    [int(end_x), int(start_y), int(start_z)],
                    [int(end_x), int(end_y), int(start_z)],
                    [int(start_x), int(end_y), int(start_z)],
                    [int(start_x), int(start_y), int(end_z)],
                    [int(end_x), int(start_y), int(end_z)],
                    [int(end_x), int(end_y), int(end_z)],
                    [int(start_x), int(end_y), int(end_z)],
                ]
            ],
            device=self.device,
            dtype=torch.long,
        ).expand(batch_size, -1, -1)

        # [x, y, z] destination
        # top-left-front, top-right-front, bottom-right-front, bottom-left-front
        # top-left-back, top-right-back, bottom-right-back, bottom-left-back
        points_dst: Tensor = tensor(
            [
                [
                    [0, 0, 0],
                    [dst_w - 1, 0, 0],
                    [dst_w - 1, dst_h - 1, 0],
                    [0, dst_h - 1, 0],
                    [0, 0, dst_d - 1],
                    [dst_w - 1, 0, dst_d - 1],
                    [dst_w - 1, dst_h - 1, dst_d - 1],
                    [0, dst_h - 1, dst_d - 1],
                ]
            ],
            device=self.device,
            dtype=torch.long,
        ).expand(batch_size, -1, -1)
        return dict(src=points_src, dst=points_dst)
