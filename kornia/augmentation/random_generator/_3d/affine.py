from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _adapted_uniform, _singular_range_check, _tuple_range_reader
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class AffineGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ```3d affine``` transformation random affine transform.

    Args:
        degrees: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal, vertical and
            depthical translations (dx,dy,dz). For example translate=(a, b, c), then
            horizontal shift will be randomly sampled in the range -img_width * a < dx < img_width * a
            vertical shift will be randomly sampled in the range -img_height * b < dy < img_height * b.
            depthical shift will be randomly sampled in the range -img_depth * c < dz < img_depth * c.
            Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If ((a, b), (c, d), (e, f)), the scale is randomly sampled from the range a <= scale_x <= b,
            c <= scale_y <= d, e <= scale_z <= f. Will keep original scale by default.
        shears: Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be applied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.

    Returns:
        A dict of parameters to be passed for transformation.
            - translations (torch.Tensor): element-wise translations with a shape of (B, 3).
            - center (torch.Tensor): element-wise center with a shape of (B, 3).
            - scale (torch.Tensor): element-wise scales with a shape of (B, 3).
            - angle (torch.Tensor): element-wise rotation angles with a shape of (B, 3).
            - sxy (torch.Tensor): element-wise x-y-facet shears with a shape of (B,).
            - sxz (torch.Tensor): element-wise x-z-facet shears with a shape of (B,).
            - syx (torch.Tensor): element-wise y-x-facet shears with a shape of (B,).
            - syz (torch.Tensor): element-wise y-z-facet shears with a shape of (B,).
            - szx (torch.Tensor): element-wise z-x-facet shears with a shape of (B,).
            - szy (torch.Tensor): element-wise z-y-facet shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        degrees: Union[
            torch.Tensor,
            float,
            Tuple[float, float],
            Tuple[float, float, float],
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ],
        translate: Optional[Union[torch.Tensor, Tuple[float, float, float]]] = None,
        scale: Optional[
            Union[
                torch.Tensor, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
            ]
        ] = None,
        shears: Union[
            torch.Tensor,
            float,
            Tuple[float, float],
            Tuple[float, float, float, float, float, float],
            Tuple[
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
            ],
        ] = None,
    ) -> None:
        super().__init__()
        self.degrees = degrees
        self.shears = shears
        self.translate = translate
        self.scale = scale

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}, shears={self.shears}, translate={self.translate}, scale={self.scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        degrees = _tuple_range_reader(self.degrees, 3, device, dtype)
        shear: Optional[torch.Tensor] = None
        if self.shears is not None:
            shear = _tuple_range_reader(self.shears, 6, device, dtype)
            self.sxy_sampler = Uniform(shear[0, 0], shear[0, 1], validate_args=False)
            self.sxz_sampler = Uniform(shear[1, 0], shear[1, 1], validate_args=False)
            self.syx_sampler = Uniform(shear[2, 0], shear[2, 1], validate_args=False)
            self.syz_sampler = Uniform(shear[3, 0], shear[3, 1], validate_args=False)
            self.szx_sampler = Uniform(shear[4, 0], shear[4, 1], validate_args=False)
            self.szy_sampler = Uniform(shear[5, 0], shear[5, 1], validate_args=False)

        # check translation range
        self._translate: Optional[torch.Tensor] = None
        if self.translate is not None:
            self._translate = torch.as_tensor(self.translate, device=device, dtype=dtype)
            _singular_range_check(self._translate, 'translate', bounds=(0, 1), mode='3d')

        # check scale range
        self._scale: Optional[torch.Tensor] = None
        if self.scale is not None:
            _scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
            if _scale.shape == torch.Size([2]):
                self._scale = _scale.unsqueeze(0).repeat(3, 1)
            elif _scale.shape != torch.Size([3, 2]):
                raise ValueError(f"'scale' shall be either shape (2) or (3, 2). Got {self.scale}.")
            else:
                self._scale = _scale
            _singular_range_check(self._scale[0], 'scale-x', bounds=(0, float('inf')), mode='2d')
            _singular_range_check(self._scale[1], 'scale-y', bounds=(0, float('inf')), mode='2d')
            _singular_range_check(self._scale[2], 'scale-z', bounds=(0, float('inf')), mode='2d')
            self.scale_1_sampler = Uniform(self._scale[0, 0], self._scale[0, 1], validate_args=False)
            self.scale_2_sampler = Uniform(self._scale[1, 0], self._scale[1, 1], validate_args=False)
            self.scale_3_sampler = Uniform(self._scale[2, 0], self._scale[2, 1], validate_args=False)

        self.yaw_sampler = Uniform(degrees[0][0], degrees[0][1], validate_args=False)
        self.pitch_sampler = Uniform(degrees[1][0], degrees[1][1], validate_args=False)
        self.roll_sampler = Uniform(degrees[2][0], degrees[2][1], validate_args=False)

        self.uniform_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        depth = batch_shape[-3]
        height = batch_shape[-2]
        width = batch_shape[-1]

        if not (
            type(depth) is int and depth > 0 and type(height) is int and height > 0 and type(width) is int and width > 0
        ):
            raise AssertionError(f"'depth', 'height' and 'width' must be integers. Got {depth}, {height}, {width}.")

        _device, _dtype = _extract_device_dtype([self.degrees, self.translate, self.scale, self.shears])

        # degrees = degrees.to(device=device, dtype=dtype)
        yaw = _adapted_rsampling((batch_size,), self.yaw_sampler, same_on_batch)
        pitch = _adapted_rsampling((batch_size,), self.pitch_sampler, same_on_batch)
        roll = _adapted_rsampling((batch_size,), self.roll_sampler, same_on_batch)
        angles = torch.stack([yaw, pitch, roll], dim=1)

        # compute tensor ranges
        if self._scale is not None:
            scale = torch.stack(
                [
                    _adapted_rsampling((batch_size,), self.scale_1_sampler, same_on_batch),
                    _adapted_rsampling((batch_size,), self.scale_2_sampler, same_on_batch),
                    _adapted_rsampling((batch_size,), self.scale_3_sampler, same_on_batch),
                ],
                dim=1,
            )
        else:
            scale = torch.ones(batch_size, device=_device, dtype=_dtype).reshape(batch_size, 1).repeat(1, 3)

        if self._translate is not None:
            max_dx: torch.Tensor = self._translate[0] * width
            max_dy: torch.Tensor = self._translate[1] * height
            max_dz: torch.Tensor = self._translate[2] * depth
            # translations should be in x,y,z
            translations = torch.stack(
                [
                    (_adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch) - 0.5) * max_dx * 2,
                    (_adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch) - 0.5) * max_dy * 2,
                    (_adapted_rsampling((batch_size,), self.uniform_sampler, same_on_batch) - 0.5) * max_dz * 2,
                ],
                dim=1,
            )
        else:
            translations = torch.zeros((batch_size, 3), device=_device, dtype=_dtype)

        # center should be in x,y,z
        center: torch.Tensor = torch.tensor([width, height, depth], device=_device, dtype=_dtype).view(1, 3) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        if self.shears is not None:
            sxy = _adapted_rsampling((batch_size,), self.sxy_sampler, same_on_batch)
            sxz = _adapted_rsampling((batch_size,), self.sxz_sampler, same_on_batch)
            syx = _adapted_rsampling((batch_size,), self.syx_sampler, same_on_batch)
            syz = _adapted_rsampling((batch_size,), self.syz_sampler, same_on_batch)
            szx = _adapted_rsampling((batch_size,), self.szx_sampler, same_on_batch)
            szy = _adapted_rsampling((batch_size,), self.szy_sampler, same_on_batch)
        else:
            sxy = sxz = syx = syz = szx = szy = torch.tensor([0] * batch_size, device=_device, dtype=_dtype)

        return dict(
            translations=torch.as_tensor(translations, device=_device, dtype=_dtype),
            center=torch.as_tensor(center, device=_device, dtype=_dtype),
            scale=torch.as_tensor(scale, device=_device, dtype=_dtype),
            angles=torch.as_tensor(angles, device=_device, dtype=_dtype),
            sxy=torch.as_tensor(sxy, device=_device, dtype=_dtype),
            sxz=torch.as_tensor(sxz, device=_device, dtype=_dtype),
            syx=torch.as_tensor(syx, device=_device, dtype=_dtype),
            syz=torch.as_tensor(syz, device=_device, dtype=_dtype),
            szx=torch.as_tensor(szx, device=_device, dtype=_dtype),
            szy=torch.as_tensor(szy, device=_device, dtype=_dtype),
        )


@_deprecated(replace_with=AffineGenerator3D.__name__)
def random_affine_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shears: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```3d affine``` transformation random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        degrees (torch.Tensor): Ranges of degrees with shape (3, 2) for yaw, pitch and roll.
        translate (torch.Tensor, optional):  maximum absolute fraction with shape (3,) for horizontal, vertical
            and depthical translations (dx,dy,dz). Will not translate by default.
        scale (torch.Tensor, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            Shaped as (6, 2) for 6 facet (xy, xz, yx, yz, zx, zy).
            The shear to the i-th facet in the range (-shear[i, 0], shear[i, 1]) will be applied.
        same_on_batch (bool): apply the same transformation across the batch. Default: False

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - translations (torch.Tensor): element-wise translations with a shape of (B, 3).
            - center (torch.Tensor): element-wise center with a shape of (B, 3).
            - scale (torch.Tensor): element-wise scales with a shape of (B, 3).
            - angle (torch.Tensor): element-wise rotation angles with a shape of (B, 3).
            - sxy (torch.Tensor): element-wise x-y-facet shears with a shape of (B,).
            - sxz (torch.Tensor): element-wise x-z-facet shears with a shape of (B,).
            - syx (torch.Tensor): element-wise y-x-facet shears with a shape of (B,).
            - syz (torch.Tensor): element-wise y-z-facet shears with a shape of (B,).
            - szx (torch.Tensor): element-wise z-x-facet shears with a shape of (B,).
            - szy (torch.Tensor): element-wise z-y-facet shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    if not (
        type(depth) is int and depth > 0 and type(height) is int and height > 0 and type(width) is int and width > 0
    ):
        raise AssertionError(f"'depth', 'height' and 'width' must be integers. Got {depth}, {height}, {width}.")

    _device, _dtype = _extract_device_dtype([degrees, translate, scale, shears])
    if degrees.shape != torch.Size([3, 2]):
        raise AssertionError(f"'degrees' must be the shape of (3, 2). Got {degrees.shape}.")
    degrees = degrees.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)
    angles = torch.stack([yaw, pitch, roll], dim=1)

    # compute tensor ranges
    if scale is not None:
        if scale.shape != torch.Size([3, 2]):
            raise AssertionError(f"'scale' must be the shape of (3, 2). Got {scale.shape}.")
        scale = scale.to(device=device, dtype=dtype)
        scale = torch.stack(
            [
                _adapted_uniform((batch_size,), scale[0, 0], scale[0, 1], same_on_batch),
                _adapted_uniform((batch_size,), scale[1, 0], scale[1, 1], same_on_batch),
                _adapted_uniform((batch_size,), scale[2, 0], scale[2, 1], same_on_batch),
            ],
            dim=1,
        )
    else:
        scale = torch.ones(batch_size, device=device, dtype=dtype).reshape(batch_size, 1).repeat(1, 3)

    if translate is not None:
        if translate.shape != torch.Size([3]):
            raise AssertionError(f"'translate' must be the shape of (2). Got {translate.shape}.")
        translate = translate.to(device=device, dtype=dtype)
        max_dx: torch.Tensor = translate[0] * width
        max_dy: torch.Tensor = translate[1] * height
        max_dz: torch.Tensor = translate[2] * depth
        # translations should be in x,y,z
        translations = torch.stack(
            [
                _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
                _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch),
                _adapted_uniform((batch_size,), -max_dz, max_dz, same_on_batch),
            ],
            dim=1,
        )
    else:
        translations = torch.zeros((batch_size, 3), device=device, dtype=dtype)

    # center should be in x,y,z
    center: torch.Tensor = torch.tensor([width, height, depth], device=device, dtype=dtype).view(1, 3) / 2.0 - 0.5
    center = center.expand(batch_size, -1)

    if shears is not None:
        if shears.shape != torch.Size([6, 2]):
            raise AssertionError(f"'shears' must be the shape of (6, 2). Got {shears.shape}.")
        shears = shears.to(device=device, dtype=dtype)
        sxy = _adapted_uniform((batch_size,), shears[0, 0], shears[0, 1], same_on_batch)
        sxz = _adapted_uniform((batch_size,), shears[1, 0], shears[1, 1], same_on_batch)
        syx = _adapted_uniform((batch_size,), shears[2, 0], shears[2, 1], same_on_batch)
        syz = _adapted_uniform((batch_size,), shears[3, 0], shears[3, 1], same_on_batch)
        szx = _adapted_uniform((batch_size,), shears[4, 0], shears[4, 1], same_on_batch)
        szy = _adapted_uniform((batch_size,), shears[5, 0], shears[5, 1], same_on_batch)
    else:
        sxy = sxz = syx = syz = szx = szy = torch.tensor([0] * batch_size, device=device, dtype=dtype)

    return dict(
        translations=translations.to(device=_device, dtype=_dtype),
        center=center.to(device=_device, dtype=_dtype),
        scale=scale.to(device=_device, dtype=_dtype),
        angles=angles.to(device=_device, dtype=_dtype),
        sxy=sxy.to(device=_device, dtype=_dtype),
        sxz=sxz.to(device=_device, dtype=_dtype),
        syx=syx.to(device=_device, dtype=_dtype),
        syz=syz.to(device=_device, dtype=_dtype),
        szx=szx.to(device=_device, dtype=_dtype),
        szy=szy.to(device=_device, dtype=_dtype),
    )
