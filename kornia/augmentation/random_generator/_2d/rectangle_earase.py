from __future__ import annotations

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _adapted_uniform, _common_param_check, _joint_range_check
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class RectangleEraseGenerator(RandomGeneratorBase):
    r"""Get parameters for ```erasing``` transformation for erasing transform.

    Args:
        scale (torch.Tensor): range of size of the origin size cropped. Shape (2).
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped. Shape (2).
        value (float): value to be filled in the erased area.

    Returns:
        A dict of parameters to be passed for transformation.
            - widths (torch.Tensor): element-wise erasing widths with a shape of (B,).
            - heights (torch.Tensor): element-wise erasing heights with a shape of (B,).
            - xs (torch.Tensor): element-wise erasing x coordinates with a shape of (B,).
            - ys (torch.Tensor): element-wise erasing y coordinates with a shape of (B,).
            - values (torch.Tensor): element-wise filling values with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        scale: torch.Tensor | tuple[float, float] = (0.02, 0.33),
        ratio: torch.Tensor | tuple[float, float] = (0.3, 3.3),
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
        scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
        ratio = torch.as_tensor(self.ratio, device=device, dtype=dtype)

        if not (isinstance(self.value, (int, float)) and self.value >= 0 and self.value <= 1):
            raise AssertionError(f"'value' must be a number between 0 - 1. Got {self.value}.")
        _joint_range_check(scale, 'scale', bounds=(0, float('inf')))
        _joint_range_check(ratio, 'ratio', bounds=(0, float('inf')))

        self.scale_sampler = Uniform(scale[0], scale[1], validate_args=False)

        if ratio[0] < 1.0 and ratio[1] > 1.0:
            self.ratio_sampler1 = Uniform(ratio[0], 1, validate_args=False)
            self.ratio_sampler2 = Uniform(1, ratio[1], validate_args=False)
            self.index_sampler = Uniform(
                torch.tensor(0, device=device, dtype=dtype),
                torch.tensor(1, device=device, dtype=dtype),
                validate_args=False,
            )
        else:
            self.ratio_sampler = Uniform(ratio[0], ratio[1], validate_args=False)
        self.uniform_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        if not (type(height) is int and height > 0 and type(width) is int and width > 0):
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
            aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
        else:
            aspect_ratios = _adapted_rsampling((batch_size,), self.ratio_sampler, same_on_batch)

        aspect_ratios = aspect_ratios.to(device=_device, dtype=_dtype)

        # based on target areas and aspect ratios, rectangle params are computed
        heights = torch.min(
            torch.max(
                torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.0, device=_device, dtype=_dtype)
            ),
            torch.tensor(height, device=_device, dtype=_dtype),
        )

        widths = torch.min(
            torch.max(
                torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.0, device=_device, dtype=_dtype)
            ),
            torch.tensor(width, device=_device, dtype=_dtype),
        )

        xs_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )
        ys_ratio = _adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch).to(
            device=_device, dtype=_dtype
        )

        xs = xs_ratio * (width - widths + 1)
        ys = ys_ratio * (height - heights + 1)

        return dict(
            widths=widths.floor(),
            heights=heights.floor(),
            xs=xs.floor(),
            ys=ys.floor(),
            values=torch.tensor([self.value] * batch_size, device=_device, dtype=_dtype),
        )


@_deprecated(replace_with=RectangleEraseGenerator.__name__)
def random_rectangles_params_generator(
    batch_size: int,
    height: int,
    width: int,
    scale: torch.Tensor,
    ratio: torch.Tensor,
    value: float = 0.0,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Get parameters for ```erasing``` transformation for erasing transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        scale (torch.Tensor): range of size of the origin size cropped. Shape (2).
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped. Shape (2).
        value (float): value to be filled in the erased area.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - widths (torch.Tensor): element-wise erasing widths with a shape of (B,).
            - heights (torch.Tensor): element-wise erasing heights with a shape of (B,).
            - xs (torch.Tensor): element-wise erasing x coordinates with a shape of (B,).
            - ys (torch.Tensor): element-wise erasing y coordinates with a shape of (B,).
            - values (torch.Tensor): element-wise filling values with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([ratio, scale])
    if not (type(height) is int and height > 0 and type(width) is int and width > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
    if not (isinstance(value, (int, float)) and value >= 0 and value <= 1):
        raise AssertionError(f"'value' must be a number between 0 - 1. Got {value}.")
    _joint_range_check(scale, 'scale', bounds=(0, float('inf')))
    _joint_range_check(ratio, 'ratio', bounds=(0, float('inf')))

    images_area = height * width
    target_areas = (
        _adapted_uniform(
            (batch_size,),
            scale[0].to(device=device, dtype=dtype),
            scale[1].to(device=device, dtype=dtype),
            same_on_batch,
        )
        * images_area
    )

    if ratio[0] < 1.0 and ratio[1] > 1.0:
        aspect_ratios1 = _adapted_uniform((batch_size,), ratio[0].to(device=device, dtype=dtype), 1, same_on_batch)
        aspect_ratios2 = _adapted_uniform((batch_size,), 1, ratio[1].to(device=device, dtype=dtype), same_on_batch)
        if same_on_batch:
            rand_idxs = (
                torch.round(
                    _adapted_uniform(
                        (1,),
                        torch.tensor(0, device=device, dtype=dtype),
                        torch.tensor(1, device=device, dtype=dtype),
                        same_on_batch,
                    )
                )
                .repeat(batch_size)
                .bool()
            )
        else:
            rand_idxs = torch.round(
                _adapted_uniform(
                    (batch_size,),
                    torch.tensor(0, device=device, dtype=dtype),
                    torch.tensor(1, device=device, dtype=dtype),
                    same_on_batch,
                )
            ).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = _adapted_uniform(
            (batch_size,),
            ratio[0].to(device=device, dtype=dtype),
            ratio[1].to(device=device, dtype=dtype),
            same_on_batch,
        )

    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(
            torch.round((target_areas * aspect_ratios) ** (1 / 2)), torch.tensor(1.0, device=device, dtype=dtype)
        ),
        torch.tensor(height, device=device, dtype=dtype),
    )

    widths = torch.min(
        torch.max(
            torch.round((target_areas / aspect_ratios) ** (1 / 2)), torch.tensor(1.0, device=device, dtype=dtype)
        ),
        torch.tensor(width, device=device, dtype=dtype),
    )

    xs_ratio = _adapted_uniform(
        (batch_size,),
        torch.tensor(0, device=device, dtype=dtype),
        torch.tensor(1, device=device, dtype=dtype),
        same_on_batch,
    )
    ys_ratio = _adapted_uniform(
        (batch_size,),
        torch.tensor(0, device=device, dtype=dtype),
        torch.tensor(1, device=device, dtype=dtype),
        same_on_batch,
    )

    xs = xs_ratio * (torch.tensor(width, device=device, dtype=dtype) - widths + 1)
    ys = ys_ratio * (torch.tensor(height, device=device, dtype=dtype) - heights + 1)

    return dict(
        widths=widths.floor().to(device=_device, dtype=_dtype),
        heights=heights.floor().to(device=_device, dtype=_dtype),
        xs=xs.floor().to(device=_device, dtype=_dtype),
        ys=ys.floor().to(device=_device, dtype=_dtype),
        values=torch.tensor([value] * batch_size, device=_device, dtype=_dtype),
    )
