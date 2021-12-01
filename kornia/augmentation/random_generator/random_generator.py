from functools import partial
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Beta, Distribution, Uniform

from kornia.geometry.bbox import bbox_generator
from kornia.utils.helpers import _deprecated, _extract_device_dtype

from ..utils import (
    _adapted_beta,
    _adapted_rsampling,
    _adapted_sampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
    _range_bound,
)

# factor, name, center, range
ParameterBound = Tuple[Any, str, Optional[float], Optional[Tuple[float, float]]]


class _PostInitInjectionMetaClass(type):
    """To inject the ``__post_init__`` function after the creation of each instance."""
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class RandomGeneratorBase(nn.Module, metaclass=_PostInitInjectionMetaClass):
    """Base class for generating random augmentation parameters."""
    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self) -> None:
        self.set_rng_device_and_dtype()

    def set_rng_device_and_dtype(
        self, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
    ) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.make_samplers(device, dtype)

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        raise NotImplementedError


class PlainUniformGenerator(RandomGeneratorBase):
    r"""Generate random parameters that distributed uniformly.

    Args:
        *samplers: a list of tuple in a pattern of ``(factor, name, center, range)``, in which
            the factor can be a two-numbered tuple, or a ``(2,)`` shaped torch tensor. The name
            will be the corresponding key of the returning dict. The center and range must be
            both provided worked as a validator to the given factor.

    Returns:
        A dict of parameters to be passed for transformation according the number of samplers
        and the pointed returning name of each tuple.
            - ``name``: element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    Example:
        >>> _ = torch.manual_seed(44)
        >>> PlainUniformGenerator(
        ...     ((0., 1.), "factor_1", None, None),
        ...     (torch.tensor([-0.5, 0.5]), "factor_2", 0.1, (-1., 1.)),
        ... )(torch.Size([2]))
        {'factor_1': tensor([0.7196, 0.7307]), 'factor_2': tensor([ 0.3278, -0.3657])}
    """
    def __init__(self, *samplers: ParameterBound) -> None:
        super().__init__()
        self.samplers = samplers
        names = []
        for factor, name, _, _ in samplers:
            if name in names:
                raise RuntimeError(f"factor name `{name}` has already been registered. Please check the duplication.")
            names.append(name)
            if isinstance(factor, torch.nn.Parameter):
                self.register_parameter(name, factor)
            elif isinstance(factor, torch.Tensor):
                self.register_buffer(name, factor)

    def __repr__(self) -> str:
        repr = ", ".join([f"{name}={factor}" for factor, name, _, _ in self.samplers])
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.sampler_dict: Dict[str, Distribution] = {}
        for factor, name, center, bound in self.samplers:
            if center is None and bound is None:
                factor = torch.as_tensor(factor, device=device, dtype=dtype)
            elif center is None or bound is None:
                raise ValueError(f"`center` and `bound` should be both None or provided. Got {center} and {bound}.")
            else:
                factor = _range_bound(
                    factor, name, center=center, bounds=bound, device=device, dtype=dtype
                )
            self.sampler_dict.update({name: Uniform(factor[0], factor[1], validate_args=False)})

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])

        return dict({
            name: _adapted_rsampling((batch_size,), dist, same_on_batch).to(
                device=_device, dtype=_dtype) for name, dist in self.sampler_dict.items()
        })


class ProbabilityGenerator(RandomGeneratorBase):
    r"""Generate random probabilities for a batch of inputs.

    Args:
        p: probability to generate an 1-d binary mask. Default value is 0.5.

    Returns:
        A dict of parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __repr__(self) -> str:
        repr = f"p={self.p}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        p = torch.tensor(float(self.p), device=device, dtype=dtype)
        self.sampler = Bernoulli(p)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        probs_mask: torch.Tensor = _adapted_sampling((batch_size,), self.sampler, same_on_batch).bool()
        return dict(probs=probs_mask)


class AffineGenerator(RandomGeneratorBase):
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        degrees: Range of degrees to select from like (min, max).
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale: scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
            If tensor, shear is a 2x2 tensor, a x-axis shear in (shear[0][0], shear[0][1]) and y-axis shear in
            (shear[1][0], shear[1][1]) will be applied. Will not apply shear by default.

    Returns:
        A dict of parameters to be passed for transformation.
            - translations (torch.Tensor): element-wise translations with a shape of (B, 2).
            - center (torch.Tensor): element-wise center with a shape of (B, 2).
            - scale (torch.Tensor): element-wise scales with a shape of (B, 2).
            - angle (torch.Tensor): element-wise rotation angles with a shape of (B,).
            - sx (torch.Tensor): element-wise x-axis shears with a shape of (B,).
            - sy (torch.Tensor): element-wise y-axis shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __repr__(self) -> str:
        repr = f"degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        _degrees = _range_bound(self.degrees, 'degrees', 0, (-360, 360)).to(device=device, dtype=dtype)
        _translate = self.translate if self.translate is None else \
            _range_bound(self.translate, 'translate', bounds=(0, 1), check='singular').to(device=device, dtype=dtype)
        _scale: Optional[torch.Tensor] = None
        if self.scale is not None:
            if len(self.scale) == 2:
                _scale = _range_bound(
                    self.scale[:2], 'scale', bounds=(0, float('inf')), check='singular').to(device=device, dtype=dtype)
            elif len(self.scale) == 4:
                _scale = torch.cat([
                    _range_bound(self.scale[:2], 'scale_x', bounds=(0, float('inf')), check='singular'),
                    _range_bound(self.scale[2:], 'scale_y', bounds=(0, float('inf')), check='singular'),  # type:ignore
                ]).to(device=device, dtype=dtype)
            else:
                raise ValueError(f"'scale' expected to be either 2 or 4 elements. Got {self.scale}")
        _shear: Optional[torch.Tensor] = None
        if self.shear is not None:
            shear = torch.as_tensor(self.shear, device=device, dtype=dtype)
            if shear.shape == torch.Size([2, 2]):
                _shear = shear
            else:
                _shear = torch.stack([
                    _range_bound(shear if shear.dim() == 0 else shear[:2], 'shear-x', 0, (-360, 360)),
                    torch.tensor([0, 0], device=device, dtype=dtype) if shear.dim() == 0 or len(shear) == 2
                    else _range_bound(shear[2:], 'shear-y', 0, (-360, 360)),
                ])

        translate_x_sampler: Optional[Uniform] = None
        translate_y_sampler: Optional[Uniform] = None
        scale_2_sampler: Optional[Uniform] = None
        scale_4_sampler: Optional[Uniform] = None
        shear_x_sampler: Optional[Uniform] = None
        shear_y_sampler: Optional[Uniform] = None

        if _translate is not None:
            translate_x_sampler = Uniform(-_translate[0], _translate[0], validate_args=False)
            translate_y_sampler = Uniform(-_translate[1], _translate[1], validate_args=False)
        if _scale is not None:
            if len(_scale) == 2:
                scale_2_sampler = Uniform(_scale[0], _scale[1], validate_args=False)
            elif len(_scale) == 4:
                scale_2_sampler = Uniform(_scale[0], _scale[1], validate_args=False)
                scale_4_sampler = Uniform(_scale[2], _scale[3], validate_args=False)
            else:
                raise ValueError(f"'scale' expected to be either 2 or 4 elements. Got {self.scale}")
        if _shear is not None:
            _joint_range_check(cast(torch.Tensor, _shear)[0], "shear")
            _joint_range_check(cast(torch.Tensor, _shear)[1], "shear")
            shear_x_sampler = Uniform(_shear[0][0], _shear[0][1], validate_args=False)
            shear_y_sampler = Uniform(_shear[1][0], _shear[1][1], validate_args=False)

        self.degree_sampler = Uniform(_degrees[0], _degrees[1], validate_args=False)
        self.translate_x_sampler = translate_x_sampler
        self.translate_y_sampler = translate_y_sampler
        self.scale_2_sampler = scale_2_sampler
        self.scale_4_sampler = scale_4_sampler
        self.shear_x_sampler = shear_x_sampler
        self.shear_y_sampler = shear_y_sampler

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type: ignore
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.degrees, self.translate, self.scale, self.shear])
        _common_param_check(batch_size, same_on_batch)
        if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
            raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

        angle = _adapted_rsampling((batch_size,), self.degree_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        # compute tensor ranges
        if self.scale_2_sampler is not None:
            _scale = _adapted_rsampling((batch_size,), self.scale_2_sampler, same_on_batch).unsqueeze(1).repeat(1, 2)
            if self.scale_4_sampler is not None:
                _scale[:, 1] = _adapted_rsampling((batch_size,), self.scale_4_sampler, same_on_batch)
            _scale = _scale.to(device=_device, dtype=_dtype)
        else:
            _scale = torch.ones((batch_size, 2), device=_device, dtype=_dtype)

        if self.translate_x_sampler is not None and self.translate_y_sampler is not None:
            translations = torch.stack([
                _adapted_rsampling((batch_size,), self.translate_x_sampler, same_on_batch) * width,
                _adapted_rsampling((batch_size,), self.translate_y_sampler, same_on_batch) * height,
            ], dim=-1)
            translations = translations.to(device=_device, dtype=_dtype)
        else:
            translations = torch.zeros((batch_size, 2), device=_device, dtype=_dtype)

        center: torch.Tensor = torch.tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
        center = center.expand(batch_size, -1)

        if self.shear_x_sampler is not None and self.shear_y_sampler is not None:
            sx = _adapted_rsampling((batch_size,), self.shear_x_sampler, same_on_batch)
            sy = _adapted_rsampling((batch_size,), self.shear_y_sampler, same_on_batch)
            sx = sx.to(device=_device, dtype=_dtype)
            sy = sy.to(device=_device, dtype=_dtype)
        else:
            sx = sy = torch.tensor([0] * batch_size, device=_device, dtype=_dtype)

        return dict(translations=translations, center=center, scale=_scale, angle=angle, sx=sx, sy=sy)


class ColorJitterGenerator(RandomGeneratorBase):
    r"""Generate random color jiter parameters for a batch of images.

    Args:
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.

    Returns:
        A dict of parameters to be passed for transformation.
            - brightness_factor: element-wise brightness factors with a shape of (B,).
            - contrast_factor: element-wise contrast factors with a shape of (B,).
            - hue_factor: element-wise hue factors with a shape of (B,).
            - saturation_factor: element-wise saturation factors with a shape of (B,).
            - order: applying orders of the color adjustments with a shape of (4). In which,
                0 is brightness adjustment; 1 is contrast adjustment;
                2 is saturation adjustment; 3 is hue adjustment.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(
        self,
        brightness: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[torch.Tensor, float, Tuple[float, float], List[float]] = 0.0,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) -> str:
        repr = f"brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        brightness: torch.Tensor = _range_bound(
            self.brightness, 'brightness', center=1.0, bounds=(0, 2), device=device, dtype=dtype
        )
        contrast: torch.Tensor = _range_bound(
            self.contrast, 'contrast', center=1.0, device=device, dtype=dtype
        )
        saturation: torch.Tensor = _range_bound(
            self.saturation, 'saturation', center=1.0, device=device, dtype=dtype
        )
        hue: torch.Tensor = _range_bound(self.hue, 'hue', bounds=(-0.5, 0.5), device=device, dtype=dtype)

        _joint_range_check(brightness, "brightness", (0, 2))
        _joint_range_check(contrast, "contrast", (0, float('inf')))
        _joint_range_check(hue, "hue", (-0.5, 0.5))
        _joint_range_check(saturation, "saturation", (0, float('inf')))

        self.brightness_sampler = Uniform(brightness[0], brightness[1], validate_args=False)
        self.contrast_sampler = Uniform(contrast[0], contrast[1], validate_args=False)
        self.hue_sampler = Uniform(hue[0], hue[1], validate_args=False)
        self.saturation_sampler = Uniform(saturation[0], saturation[1], validate_args=False)
        self.randperm = partial(torch.randperm, device=device, dtype=dtype)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.brightness, self.contrast, self.hue, self.saturation])
        brightness_factor = _adapted_rsampling((batch_size,), self.brightness_sampler, same_on_batch)
        contrast_factor = _adapted_rsampling((batch_size,), self.contrast_sampler, same_on_batch)
        hue_factor = _adapted_rsampling((batch_size,), self.hue_sampler, same_on_batch)
        saturation_factor = _adapted_rsampling((batch_size,), self.saturation_sampler, same_on_batch)
        return dict(
            brightness_factor=brightness_factor.to(device=_device, dtype=_dtype),
            contrast_factor=contrast_factor.to(device=_device, dtype=_dtype),
            hue_factor=hue_factor.to(device=_device, dtype=_dtype),
            saturation_factor=saturation_factor.to(device=_device, dtype=_dtype),
            order=self.randperm(4).to(device=_device, dtype=_dtype).long(),
        )


class CropGenerator(RandomGeneratorBase):
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(
        self,
        size: Union[Tuple[int, int], torch.Tensor],
        resize_to: Optional[Tuple[int, int]] = None
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
        self.rand_sampler = Uniform(
            torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.size if isinstance(self.size, torch.Tensor) else None])

        if batch_size == 0:
            return dict(
                src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
                dst=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            )

        input_size = (batch_shape[-2], batch_shape[-1])
        if not isinstance(self.size, torch.Tensor):
            size = torch.tensor(self.size, device=_device, dtype=_dtype).repeat(batch_size, 1)
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
            x_start = (_adapted_rsampling(
                (batch_size,), self.rand_sampler, same_on_batch).to(x_diff) * x_diff[0]).floor()
            y_start = (_adapted_rsampling(
                (batch_size,), self.rand_sampler, same_on_batch).to(y_diff) * y_diff[0]).floor()
        else:
            x_start = (_adapted_rsampling(
                (batch_size,), self.rand_sampler, same_on_batch).to(x_diff) * x_diff).floor()
            y_start = (_adapted_rsampling(
                (batch_size,), self.rand_sampler, same_on_batch).to(y_diff) * y_diff).floor()
        crop_src = bbox_generator(
            x_start.view(-1).to(device=_device, dtype=_dtype),
            y_start.view(-1).to(device=_device, dtype=_dtype),
            torch.where(size[:, 1] == 0, torch.tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
            torch.where(size[:, 0] == 0, torch.tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]),
        )

        if self.resize_to is None:
            crop_dst = bbox_generator(
                torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
                torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
                size[:, 1],
                size[:, 0],
            )
        else:
            if not (
                len(self.resize_to) == 2
                and isinstance(self.resize_to[0], (int,))
                and isinstance(self.resize_to[1], (int,))
                and self.resize_to[0] > 0
                and self.resize_to[1] > 0
            ):
                raise AssertionError(f"`resize_to` must be a tuple of 2 positive integers. Got {self.resize_to}.")
            crop_dst = torch.tensor(
                [[
                    [0, 0],
                    [self.resize_to[1] - 1, 0],
                    [self.resize_to[1] - 1, self.resize_to[0] - 1],
                    [0, self.resize_to[0] - 1]
                ]],
                device=_device,
                dtype=_dtype,
            ).repeat(batch_size, 1, 1)

        _input_size = torch.tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

        return dict(src=crop_src, dst=crop_dst, input_size=_input_size)


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
        ratio: Union[torch.Tensor, Tuple[float, float]]
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
            torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))
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

        rand = _adapted_rsampling(
            (batch_size, 10), self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype)
        area = (rand * (self.scale[1] - self.scale[0]) + self.scale[0]) * size[0] * size[1]
        log_ratio = _adapted_rsampling(
            (batch_size, 10), self.log_ratio_sampler, same_on_batch).to(device=_device, dtype=_dtype)
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


class PerspectiveGenerator(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: the degree of distortion, ranged from 0 to 1.

    Returns:
        A dict of parameters to be passed for transformation.
            - start_points (torch.Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
            - end_points (torch.Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(self, distortion_scale: Union[torch.Tensor, float] = 0.5) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}.")
        self.rand_val_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _device, _dtype = _extract_device_dtype([self.distortion_scale])
        _common_param_check(batch_size, same_on_batch)
        if not (type(height) is int and height > 0 and type(width) is int and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        start_points: torch.Tensor = torch.tensor(
            [[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
            device=_device,
            dtype=_dtype,
        ).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = self._distortion_scale * width / 2
        fy = self._distortion_scale * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2).to(device=_device, dtype=_dtype)

        # TODO: This line somehow breaks the gradcheck
        rand_val: torch.Tensor = _adapted_rsampling(
            start_points.shape, self.rand_val_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        pts_norm = torch.tensor(
            [[[1, 1], [-1, 1], [-1, -1], [1, -1]]], device=_device, dtype=_dtype
        )

        end_points = start_points + factor * rand_val * pts_norm

        return dict(start_points=start_points, end_points=end_points)


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
        scale: Union[torch.Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[torch.Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.
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
                validate_args=False
            )
        else:
            self.ratio_sampler = Uniform(ratio[0], ratio[1], validate_args=False)
        self.uniform_sampler = Uniform(
            torch.tensor(0, device=device, dtype=dtype),
            torch.tensor(1, device=device, dtype=dtype),
            validate_args=False
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]
        if not (type(height) is int and height > 0 and type(width) is int and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.ratio, self.scale])
        images_area = height * width
        target_areas = _adapted_rsampling(
            (batch_size,), self.scale_sampler, same_on_batch).to(device=_device, dtype=_dtype) * images_area

        if self.ratio[0] < 1.0 and self.ratio[1] > 1.0:
            aspect_ratios1 = _adapted_rsampling((batch_size,), self.ratio_sampler1, same_on_batch)
            aspect_ratios2 = _adapted_rsampling((batch_size,), self.ratio_sampler2, same_on_batch)
            if same_on_batch:
                rand_idxs = torch.round(
                    _adapted_rsampling((1,), self.index_sampler, same_on_batch)).repeat(batch_size).bool()
            else:
                rand_idxs = torch.round(
                    _adapted_rsampling((batch_size,), self.index_sampler, same_on_batch)).bool()
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

        xs_ratio = _adapted_rsampling(
            (batch_size,), self.uniform_sampler, same_on_batch).to(device=_device, dtype=_dtype)
        ys_ratio = _adapted_rsampling(
            (batch_size,), self.uniform_sampler, same_on_batch).to(device=_device, dtype=_dtype)

        xs = xs_ratio * (width - widths + 1)
        ys = ys_ratio * (height - heights + 1)

        return dict(
            widths=widths.floor(),
            heights=heights.floor(),
            xs=xs.floor(),
            ys=ys.floor(),
            values=torch.tensor([self.value] * batch_size, device=_device, dtype=_dtype),
        )


class MotionBlurGenerator(RandomGeneratorBase):
    r"""Get parameters for motion blur.

    Args:
        kernel_size: motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.

    Returns:
        A dict of parameters to be passed for transformation.
            - ksize_factor (torch.Tensor): element-wise kernel size factors with a shape of (B,).
            - angle_factor (torch.Tensor): element-wise angle factors with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise direction factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        angle: Union[torch.Tensor, float, Tuple[float, float]],
        direction: Union[torch.Tensor, float, Tuple[float, float]],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        angle = _range_bound(self.angle, 'angle', center=0.0, bounds=(-360, 360)).to(device=device, dtype=dtype)
        direction = _range_bound(self.direction, 'direction', center=0.0, bounds=(-1, 1)).to(device=device, dtype=dtype)
        if isinstance(self.kernel_size, int):
            if not (self.kernel_size >= 3 and self.kernel_size % 2 == 1):
                raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {self.kernel_size}.")
            self.ksize_sampler = Uniform(self.kernel_size // 2, self.kernel_size // 2, validate_args=False)
        elif isinstance(self.kernel_size, tuple):
            # kernel_size is fixed across the batch
            if len(self.kernel_size) != 2:
                raise AssertionError(f"`kernel_size` must be (2,) if it is a tuple. Got {self.kernel_size}.")
            self.ksize_sampler = Uniform(self.kernel_size[0] // 2, self.kernel_size[1] // 2, validate_args=False)
        else:
            raise TypeError(f"Unsupported type: {type(self.kernel_size)}")

        self.angle_sampler = Uniform(angle[0], angle[1], validate_args=False)
        self.direction_sampler = Uniform(direction[0], direction[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        # self.ksize_factor.expand((batch_size, -1))
        _device, _dtype = _extract_device_dtype([self.angle, self.direction])
        angle_factor = _adapted_rsampling((batch_size,), self.angle_sampler, same_on_batch)
        direction_factor = _adapted_rsampling((batch_size,), self.direction_sampler, same_on_batch)
        ksize_factor = _adapted_rsampling((batch_size,), self.ksize_sampler, same_on_batch).int() * 2 + 1

        return dict(
            ksize_factor=ksize_factor.to(device=_device, dtype=torch.int32),
            angle_factor=angle_factor.to(device=_device, dtype=_dtype),
            direction_factor=direction_factor.to(device=_device, dtype=_dtype),
        )


class PosterizeGenerator(RandomGeneratorBase):
    r"""Generate random posterize parameters for a batch of images.

    Args:
        bits: Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8).
            If tuple (x, y), bits will be generated from (x, y).

    Returns:
        A dict of parameters to be passed for transformation.
            - bits_factor (torch.Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """
    def __init__(self, bits: Union[int, Tuple[int, int], torch.Tensor]) -> None:
        super().__init__()
        self.bits = bits

    def __repr__(self) -> str:
        repr = f"bits={self.bits}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        bits = torch.as_tensor(self.bits, device=device, dtype=dtype)
        if len(bits.size()) == 0:
            bits = bits.repeat(2)
            bits[1] = 8
        elif not (len(bits.size()) == 1 and bits.size(0) == 2):
            raise ValueError(f"'bits' shall be either a scalar or a length 2 tensor. Got {bits}.")
        _joint_range_check(bits, 'bits', (0, 8))
        self.bit_sampler = Uniform(bits[0], bits[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.bits if isinstance(self.bits, torch.Tensor) else None])
        bits_factor = _adapted_rsampling((batch_size,), self.bit_sampler, same_on_batch)
        return dict(bits_factor=bits_factor.to(device=_device, dtype=torch.int32))


class MixupGenerator(RandomGeneratorBase):
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.

    Returns:
        A dict of parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B,).
            - mixup_lambdas (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None, p: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val
        self.p = p

    def __repr__(self) -> str:
        repr = f"lambda_val={self.lambda_val}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.lambda_val is None:
            lambda_val = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            lambda_val = torch.as_tensor(self.lambda_val, device=device, dtype=dtype)

        _joint_range_check(lambda_val, 'lambda_val', bounds=(0, 1))
        self.lambda_sampler = Uniform(lambda_val[0], lambda_val[1], validate_args=False)
        self.prob_sampler = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.lambda_val])

        with torch.no_grad():
            batch_probs: torch.Tensor = _adapted_sampling((batch_size,), self.prob_sampler, same_on_batch)
        mixup_pairs: torch.Tensor = torch.randperm(batch_size, device=_device, dtype=_dtype).long()
        mixup_lambdas: torch.Tensor = _adapted_rsampling((batch_size,), self.lambda_sampler, same_on_batch)
        mixup_lambdas = mixup_lambdas * batch_probs

        return dict(
            mixup_pairs=mixup_pairs.to(device=_device, dtype=torch.long),
            mixup_lambdas=mixup_lambdas.to(device=_device, dtype=_dtype),
        )


class CutmixGenerator(RandomGeneratorBase):
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size (torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (num_mix, B).
            - crop_src (torch.Tensor): element-wise probabilities with a shape of (num_mix, B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        cut_size: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[torch.Tensor, float]] = None,
        num_mix: int = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.beta = beta
        self.num_mix = num_mix
        self.p = p

        if not (num_mix >= 1 and isinstance(num_mix, (int,))):
            raise AssertionError(f"`num_mix` must be an integer greater than 1. Got {num_mix}.")

    def __repr__(self) -> str:
        repr = f"cut_size={self.cut_size}, beta={self.beta}, num_mix={self.num_mix}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.beta is None:
            self._beta = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            self._beta = torch.as_tensor(self.beta, device=device, dtype=dtype)
        if self.cut_size is None:
            self._cut_size = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            self._cut_size = torch.as_tensor(self.cut_size, device=device, dtype=dtype)

        _joint_range_check(self._cut_size, 'cut_size', bounds=(0, 1))

        self.beta_sampler = Beta(self._beta, self._beta)
        self.prob_sampler = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))
        self.rand_sampler = Uniform(
            torch.tensor(0., device=device, dtype=dtype),
            torch.tensor(1., device=device, dtype=dtype),
            validate_args=False
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        if not (type(height) is int and height > 0 and type(width) is int and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
        _device, _dtype = _extract_device_dtype([self.beta, self.cut_size])
        _common_param_check(batch_size, same_on_batch)

        if batch_size == 0:
            return dict(
                mix_pairs=torch.zeros([0, 3], device=_device, dtype=torch.long),
                crop_src=torch.zeros([0, 4, 2], device=_device, dtype=torch.long),
            )

        with torch.no_grad():
            batch_probs: torch.Tensor = _adapted_sampling(
                (batch_size * self.num_mix,), self.prob_sampler, same_on_batch)
        mix_pairs: torch.Tensor = torch.rand(self.num_mix, batch_size, device=_device, dtype=_dtype).argsort(dim=1)
        cutmix_betas: torch.Tensor = _adapted_rsampling((batch_size * self.num_mix,), self.beta_sampler, same_on_batch)

        # Note: torch.clamp does not accept tensor, cutmix_betas.clamp(cut_size[0], cut_size[1]) throws:
        # Argument 1 to "clamp" of "_TensorBase" has incompatible type "Tensor"; expected "float"
        cutmix_betas = torch.min(torch.max(cutmix_betas, self._cut_size[0]), self._cut_size[1])
        cutmix_rate = torch.sqrt(1.0 - cutmix_betas) * batch_probs

        cut_height = (cutmix_rate * height).floor().to(device=_device, dtype=_dtype)
        cut_width = (cutmix_rate * width).floor().to(device=_device, dtype=_dtype)
        _gen_shape = (1,)

        if same_on_batch:
            _gen_shape = (cut_height.size(0),)
            cut_height = cut_height[0]
            cut_width = cut_width[0]

        # Reserve at least 1 pixel for cropping.
        x_start: torch.Tensor = _adapted_rsampling(
            _gen_shape, self.rand_sampler, same_on_batch) * (width - cut_width - 1)
        y_start: torch.Tensor = _adapted_rsampling(
            _gen_shape, self.rand_sampler, same_on_batch) * (height - cut_height - 1)
        x_start = x_start.floor().to(device=_device, dtype=_dtype)
        y_start = y_start.floor().to(device=_device, dtype=_dtype)

        crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

        # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
        crop_src = crop_src.view(self.num_mix, batch_size, 4, 2)

        return dict(
            mix_pairs=mix_pairs.to(device=_device, dtype=torch.long),
            crop_src=crop_src.floor().to(device=_device, dtype=_dtype),
        )


def random_prob_generator(
    batch_size: int,
    p: float = 0.5,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        torch.Tensor: parameters to be passed for transformation.
            - probs (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    if not isinstance(p, (int, float)) or p > 1 or p < 0:
        raise TypeError(f"The probability should be a float number within [0, 1]. Got {type(p)}.")

    _bernoulli = Bernoulli(torch.tensor(float(p), device=device, dtype=dtype))
    probs_mask: torch.Tensor = _adapted_sampling((batch_size,), _bernoulli, same_on_batch).bool()

    return probs_mask


@_deprecated(replace_with=ColorJitterGenerator.__name__)
def random_color_jitter_generator(
    batch_size: int,
    brightness: Optional[torch.Tensor] = None,
    contrast: Optional[torch.Tensor] = None,
    saturation: Optional[torch.Tensor] = None,
    hue: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (torch.Tensor, optional): Brightness factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b <= 2. Default value is [0., 0.].
        contrast (torch.Tensor, optional): Contrast factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b. Default value is [0., 0.].
        saturation (torch.Tensor, optional): Saturation factor tensor of range (a, b).
            The provided range must follow 0 <= a <= b. Default value is [0., 0.].
        hue (torch.Tensor, optional): Saturation factor tensor of range (a, b).
            The provided range must follow -0.5 <= a <= b < 0.5. Default value is [0., 0.].
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - brightness_factor (torch.Tensor): element-wise brightness factors with a shape of (B,).
            - contrast_factor (torch.Tensor): element-wise contrast factors with a shape of (B,).
            - hue_factor (torch.Tensor): element-wise hue factors with a shape of (B,).
            - saturation_factor (torch.Tensor): element-wise saturation factors with a shape of (B,).
            - order (torch.Tensor): applying orders of the color adjustments with a shape of (4). In which,
                0 is brightness adjustment; 1 is contrast adjustment;
                2 is saturation adjustment; 3 is hue adjustment.

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([brightness, contrast, hue, saturation])
    brightness = torch.as_tensor([0.0, 0.0] if brightness is None else brightness, device=device, dtype=dtype)
    contrast = torch.as_tensor([0.0, 0.0] if contrast is None else contrast, device=device, dtype=dtype)
    hue = torch.as_tensor([0.0, 0.0] if hue is None else hue, device=device, dtype=dtype)
    saturation = torch.as_tensor([0.0, 0.0] if saturation is None else saturation, device=device, dtype=dtype)

    _joint_range_check(brightness, "brightness", (0, 2))
    _joint_range_check(contrast, "contrast", (0, float('inf')))
    _joint_range_check(hue, "hue", (-0.5, 0.5))
    _joint_range_check(saturation, "saturation", (0, float('inf')))

    brightness_factor = _adapted_uniform((batch_size,), brightness[0], brightness[1], same_on_batch)
    contrast_factor = _adapted_uniform((batch_size,), contrast[0], contrast[1], same_on_batch)
    hue_factor = _adapted_uniform((batch_size,), hue[0], hue[1], same_on_batch)
    saturation_factor = _adapted_uniform((batch_size,), saturation[0], saturation[1], same_on_batch)

    return dict(
        brightness_factor=brightness_factor.to(device=_device, dtype=_dtype),
        contrast_factor=contrast_factor.to(device=_device, dtype=_dtype),
        hue_factor=hue_factor.to(device=_device, dtype=_dtype),
        saturation_factor=saturation_factor.to(device=_device, dtype=_dtype),
        order=torch.randperm(4, device=_device, dtype=_dtype).long(),
    )


@_deprecated(replace_with=PerspectiveGenerator.__name__)
def random_perspective_generator(
    batch_size: int,
    height: int,
    width: int,
    distortion_scale: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - start_points (torch.Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
            - end_points (torch.Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    if not (distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1):
        raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}.")
    if not (type(height) is int and height > 0 and type(width) is int and width > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

    start_points: torch.Tensor = torch.tensor(
        [[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
        device=distortion_scale.device,
        dtype=distortion_scale.dtype,
    ).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx = distortion_scale * width / 2
    fy = distortion_scale * height / 2

    factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

    # TODO: This line somehow breaks the gradcheck
    rand_val: torch.Tensor = _adapted_uniform(
        start_points.shape,
        torch.tensor(0, device=device, dtype=dtype),
        torch.tensor(1, device=device, dtype=dtype),
        same_on_batch,
    ).to(device=distortion_scale.device, dtype=distortion_scale.dtype)

    pts_norm = torch.tensor(
        [[[1, 1], [-1, 1], [-1, -1], [1, -1]]], device=distortion_scale.device, dtype=distortion_scale.dtype
    )
    end_points = start_points + factor * rand_val * pts_norm

    return dict(start_points=start_points, end_points=end_points)


@_deprecated(replace_with=AffineGenerator.__name__)
def random_affine_generator(
    batch_size: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shear: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        degrees (torch.Tensor): Range of degrees to select from like (min, max).
        translate (tensor, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tensor, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (tensor, optional): Range of degrees to select from.
            Shear is a 2x2 tensor, a x-axis shear in (shear[0][0], shear[0][1]) and y-axis shear in
            (shear[1][0], shear[1][1]) will be applied. Will not apply shear by default.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - translations (torch.Tensor): element-wise translations with a shape of (B, 2).
            - center (torch.Tensor): element-wise center with a shape of (B, 2).
            - scale (torch.Tensor): element-wise scales with a shape of (B, 2).
            - angle (torch.Tensor): element-wise rotation angles with a shape of (B,).
            - sx (torch.Tensor): element-wise x-axis shears with a shape of (B,).
            - sy (torch.Tensor): element-wise y-axis shears with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(degrees, "degrees")
    if not (isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0):
        raise AssertionError(f"`width` and `height` must be positive integers. Got {width}, {height}.")

    _device, _dtype = _extract_device_dtype([degrees, translate, scale, shear])
    degrees = degrees.to(device=device, dtype=dtype)
    angle = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)
    angle = angle.to(device=_device, dtype=_dtype)

    # compute tensor ranges
    if scale is not None:
        scale = scale.to(device=device, dtype=dtype)
        if not (len(scale.shape) == 1 and len(scale) in (2, 4)):
            raise AssertionError(f"`scale` shall have 2 or 4 elements. Got {scale}.")
        _joint_range_check(cast(torch.Tensor, scale[:2]), "scale")
        _scale = _adapted_uniform((batch_size,), scale[0], scale[1], same_on_batch).unsqueeze(1).repeat(1, 2)
        if len(scale) == 4:
            _joint_range_check(cast(torch.Tensor, scale[2:]), "scale_y")
            _scale[:, 1] = _adapted_uniform((batch_size,), scale[2], scale[3], same_on_batch)
        _scale = _scale.to(device=_device, dtype=_dtype)
    else:
        _scale = torch.ones((batch_size, 2), device=_device, dtype=_dtype)

    if translate is not None:
        translate = translate.to(device=device, dtype=dtype)
        if not (0.0 <= translate[0] <= 1.0 and 0.0 <= translate[1] <= 1.0 and translate.shape == torch.Size([2])):
            raise AssertionError(f"Expect translate contains two elements and ranges are in [0, 1]. Got {translate}.")
        max_dx: torch.Tensor = translate[0] * width
        max_dy: torch.Tensor = translate[1] * height
        translations = torch.stack(
            [
                _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
                _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch),
            ],
            dim=-1,
        )
        translations = translations.to(device=_device, dtype=_dtype)
    else:
        translations = torch.zeros((batch_size, 2), device=_device, dtype=_dtype)

    center: torch.Tensor = torch.tensor([width, height], device=_device, dtype=_dtype).view(1, 2) / 2.0 - 0.5
    center = center.expand(batch_size, -1)

    if shear is not None:
        shear = shear.to(device=device, dtype=dtype)
        _joint_range_check(cast(torch.Tensor, shear)[0], "shear")
        _joint_range_check(cast(torch.Tensor, shear)[1], "shear")
        sx = _adapted_uniform((batch_size,), shear[0][0], shear[0][1], same_on_batch)
        sy = _adapted_uniform((batch_size,), shear[1][0], shear[1][1], same_on_batch)
        sx = sx.to(device=_device, dtype=_dtype)
        sy = sy.to(device=_device, dtype=_dtype)
    else:
        sx = sy = torch.tensor([0] * batch_size, device=_device, dtype=_dtype)

    return dict(translations=translations, center=center, scale=_scale, angle=angle, sx=sx, sy=sy)


@_deprecated()
def random_rotation_generator(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): range of degrees with shape (2) to select from.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - degrees (torch.Tensor): element-wise rotation degrees with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(degrees, "degrees")

    _degrees = _adapted_uniform(
        (batch_size,),
        degrees[0].to(device=device, dtype=dtype),
        degrees[1].to(device=device, dtype=dtype),
        same_on_batch,
    )
    _degrees = _degrees.to(device=degrees.device, dtype=degrees.dtype)

    return dict(degrees=_degrees)


@_deprecated(replace_with=CropGenerator.__name__)
def random_crop_generator(
    batch_size: int,
    input_size: Tuple[int, int],
    size: Union[Tuple[int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int]] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (h, w).
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> crop_size = torch.tensor([[25, 28], [27, 29], [26, 28]])
        >>> random_crop_generator(3, (30, 30), size=crop_size, same_on_batch=False)
        {'src': tensor([[[ 1.,  0.],
                 [28.,  0.],
                 [28., 24.],
                 [ 1., 24.]],
        <BLANKLINE>
                [[ 1.,  1.],
                 [29.,  1.],
                 [29., 27.],
                 [ 1., 27.]],
        <BLANKLINE>
                [[ 0.,  3.],
                 [27.,  3.],
                 [27., 28.],
                 [ 0., 28.]]]), 'dst': tensor([[[ 0.,  0.],
                 [27.,  0.],
                 [27., 24.],
                 [ 0., 24.]],
        <BLANKLINE>
                [[ 0.,  0.],
                 [28.,  0.],
                 [28., 26.],
                 [ 0., 26.]],
        <BLANKLINE>
                [[ 0.,  0.],
                 [27.,  0.],
                 [27., 25.],
                 [ 0., 25.]]]), 'input_size': tensor([[30, 30],
                [30, 30],
                [30, 30]])}
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([size if isinstance(size, torch.Tensor) else None])
    # Use float point instead
    _dtype = _dtype if _dtype in [torch.float16, torch.float32, torch.float64] else dtype
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=_device, dtype=_dtype).repeat(batch_size, 1)
    else:
        size = size.to(device=_device, dtype=_dtype)
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

    if batch_size == 0:
        return dict(
            src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            dst=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
        )

    if same_on_batch:
        # If same_on_batch, select the first then repeat.
        x_start = _adapted_uniform((batch_size,), 0, x_diff[0].to(device=device, dtype=dtype), same_on_batch).floor()
        y_start = _adapted_uniform((batch_size,), 0, y_diff[0].to(device=device, dtype=dtype), same_on_batch).floor()
    else:
        x_start = _adapted_uniform((1,), 0, x_diff.to(device=device, dtype=dtype), same_on_batch).floor()
        y_start = _adapted_uniform((1,), 0, y_diff.to(device=device, dtype=dtype), same_on_batch).floor()
    crop_src = bbox_generator(
        x_start.view(-1).to(device=_device, dtype=_dtype),
        y_start.view(-1).to(device=_device, dtype=_dtype),
        torch.where(size[:, 1] == 0, torch.tensor(input_size[1], device=_device, dtype=_dtype), size[:, 1]),
        torch.where(size[:, 0] == 0, torch.tensor(input_size[0], device=_device, dtype=_dtype), size[:, 0]),
    )

    if resize_to is None:
        crop_dst = bbox_generator(
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            size[:, 1],
            size[:, 0],
        )
    else:
        if not (
            len(resize_to) == 2
            and isinstance(resize_to[0], (int,))
            and isinstance(resize_to[1], (int,))
            and resize_to[0] > 0
            and resize_to[1] > 0
        ):
            raise AssertionError(f"`resize_to` must be a tuple of 2 positive integers. Got {resize_to}.")
        crop_dst = torch.tensor(
            [[[0, 0], [resize_to[1] - 1, 0], [resize_to[1] - 1, resize_to[0] - 1], [0, resize_to[0] - 1]]],
            device=_device,
            dtype=_dtype,
        ).repeat(batch_size, 1, 1)

    _input_size = torch.tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

    return dict(src=crop_src, dst=crop_dst, input_size=_input_size)


@_deprecated()
def random_crop_size_generator(
    batch_size: int,
    size: Tuple[int, int],
    scale: torch.Tensor,
    ratio: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        batch_size (int): the tensor batch size.
        size (Tuple[int, int]): expected output size of each edge.
        scale (torch.Tensor): range of size of the origin size cropped with (2,) shape.
        ratio (torch.Tensor): range of aspect ratio of the origin aspect ratio cropped with (2,) shape.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

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
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(scale, "scale")
    _joint_range_check(ratio, "ratio")
    if not (len(size) == 2 and type(size[0]) is int and size[1] > 0 and type(size[1]) is int and size[1] > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {size}.")

    _device, _dtype = _extract_device_dtype([scale, ratio])

    if batch_size == 0:
        return dict(size=torch.zeros([0, 2], device=_device, dtype=_dtype))

    scale = scale.to(device=device, dtype=dtype)
    ratio = ratio.to(device=device, dtype=dtype)
    # 10 trails for each element
    area = _adapted_uniform((batch_size, 10), scale[0] * size[0] * size[1], scale[1] * size[0] * size[1], same_on_batch)
    log_ratio = _adapted_uniform((batch_size, 10), torch.log(ratio[0]), torch.log(ratio[1]), same_on_batch)
    aspect_ratio = torch.exp(log_ratio)

    w = torch.sqrt(area * aspect_ratio).round().floor()
    h = torch.sqrt(area / aspect_ratio).round().floor()
    # Element-wise w, h condition
    cond = ((0 < w) * (w < size[0]) * (0 < h) * (h < size[1])).int()

    # torch.argmax is not reproducible across devices: https://github.com/pytorch/pytorch/issues/17738
    # Here, we will select the first occurrence of the duplicated elements.
    cond_bool, argmax_dim1 = ((cond.cumsum(1) == 1) & cond.bool()).max(1)
    h_out = w[torch.arange(0, batch_size, device=device, dtype=torch.long), argmax_dim1]
    w_out = h[torch.arange(0, batch_size, device=device, dtype=torch.long), argmax_dim1]

    if not cond_bool.all():
        # Fallback to center crop
        in_ratio = float(size[0]) / float(size[1])
        if in_ratio < ratio.min():
            h_ct = torch.tensor(size[0], device=device, dtype=dtype)
            w_ct = torch.round(h_ct / ratio.min())
        elif in_ratio > ratio.min():
            w_ct = torch.tensor(size[1], device=device, dtype=dtype)
            h_ct = torch.round(w_ct * ratio.min())
        else:  # whole image
            h_ct = torch.tensor(size[0], device=device, dtype=dtype)
            w_ct = torch.tensor(size[1], device=device, dtype=dtype)
        h_ct = h_ct.floor()
        w_ct = w_ct.floor()

        h_out = h_out.where(cond_bool, h_ct)
        w_out = w_out.where(cond_bool, w_ct)

    return dict(size=torch.stack([h_out, w_out], dim=1).to(device=_device, dtype=_dtype))


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
) -> Dict[str, torch.Tensor]:
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


def center_crop_generator(
    batch_size: int, height: int, width: int, size: Tuple[int, int], device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```center_crop``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (h, w).
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 4, 2).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 4, 2).

    Note:
        No random number will be generated.
    """
    _common_param_check(batch_size)
    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")
    if not (type(height) is int and height > 0 and type(width) is int and width > 0):
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
    points_src: torch.Tensor = torch.tensor(
        [[[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]], device=device, dtype=torch.long
    ).expand(batch_size, -1, -1)

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], device=device, dtype=torch.long
    ).expand(batch_size, -1, -1)

    _input_size = torch.tensor((height, width), device=device, dtype=torch.long).expand(batch_size, -1)

    return dict(src=points_src, dst=points_dst, input_size=_input_size)


@_deprecated(replace_with=MotionBlurGenerator.__name__)
def random_motion_blur_generator(
    batch_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    angle: torch.Tensor,
    direction: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for motion blur.

    Args:
        batch_size (int): the tensor batch size.
        kernel_size (int or (int, int)): motion kernel size (odd and positive) or range.
        angle (torch.Tensor): angle of the motion blur in degrees (anti-clockwise rotation).
        direction (torch.Tensor): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with
            angle provided via angle), while higher values towards 1.0 will point the motion
            blur forward. A value of 0.0 leads to a uniformly (but still angled) motion blur.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - ksize_factor (torch.Tensor): element-wise kernel size factors with a shape of (B,).
            - angle_factor (torch.Tensor): element-wise angle factors with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise direction factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(angle, 'angle')
    _joint_range_check(direction, 'direction', (-1, 1))

    _device, _dtype = _extract_device_dtype([angle, direction])

    if isinstance(kernel_size, int):
        if not (kernel_size >= 3 and kernel_size % 2 == 1):
            raise AssertionError(f"`kernel_size` must be odd and greater than 3. Got {kernel_size}.")
        ksize_factor = torch.tensor([kernel_size] * batch_size, device=device, dtype=dtype)
    elif isinstance(kernel_size, tuple):
        # kernel_size is fixed across the batch
        if len(kernel_size) != 2:
            raise AssertionError(f"`kernel_size` must be (2,) if it is a tuple. Got {kernel_size}.")
        ksize_factor = (
            _adapted_uniform((batch_size,), kernel_size[0] // 2, kernel_size[1] // 2, same_on_batch=True).int() * 2 + 1
        )
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    angle_factor = _adapted_uniform(
        (batch_size,), angle[0].to(device=device, dtype=dtype), angle[1].to(device=device, dtype=dtype), same_on_batch
    )

    direction_factor = _adapted_uniform(
        (batch_size,),
        direction[0].to(device=device, dtype=dtype),
        direction[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    return dict(
        ksize_factor=ksize_factor.to(device=_device, dtype=torch.int32),
        angle_factor=angle_factor.to(device=_device, dtype=_dtype),
        direction_factor=direction_factor.to(device=_device, dtype=_dtype),
    )


@_deprecated()
def random_solarize_generator(
    batch_size: int,
    thresholds: torch.Tensor = torch.tensor([0.4, 0.6]),
    additions: torch.Tensor = torch.tensor([-0.1, 0.1]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate random solarize parameters for a batch of images.

    For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the pixel value
    to be between 0 and 1.0

    Args:
        batch_size (int): the number of images.
        thresholds (torch.Tensor): Pixels less than threshold will selected. Otherwise, subtract 1.0 from the pixel.
            Takes in a range tensor of (0, 1). Default value will be sampled from [0.4, 0.6].
        additions (torch.Tensor): The value is between -0.5 and 0.5. Default value will be sampled from [-0.1, 0.1]
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - thresholds_factor (torch.Tensor): element-wise thresholds factors with a shape of (B,).
            - additions_factor (torch.Tensor): element-wise additions factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(thresholds, 'thresholds', (0, 1))
    _joint_range_check(additions, 'additions', (-0.5, 0.5))

    _device, _dtype = _extract_device_dtype([thresholds, additions])

    thresholds_factor = _adapted_uniform(
        (batch_size,),
        thresholds[0].to(device=device, dtype=dtype),
        thresholds[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    additions_factor = _adapted_uniform(
        (batch_size,),
        additions[0].to(device=device, dtype=dtype),
        additions[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    return dict(
        thresholds_factor=thresholds_factor.to(device=_device, dtype=_dtype),
        additions_factor=additions_factor.to(device=_device, dtype=_dtype),
    )


@_deprecated()
def random_posterize_generator(
    batch_size: int,
    bits: torch.Tensor = torch.tensor([3, 5]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate random posterize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        bits (int or tuple): Takes in an integer tuple tensor that ranged from 0 ~ 8. Default value is [3, 5].
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - bits_factor (torch.Tensor): element-wise bit factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(bits, 'bits', (0, 8))
    bits_factor = _adapted_uniform(
        (batch_size,), bits[0].to(device=device, dtype=dtype), bits[1].to(device=device, dtype=dtype), same_on_batch
    ).int()

    return dict(bits_factor=bits_factor.to(device=bits.device, dtype=torch.int32))


@_deprecated()
def random_sharpness_generator(
    batch_size: int,
    sharpness: torch.Tensor = torch.tensor([0, 1.0]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate random sharpness parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        sharpness (torch.Tensor): Must be above 0. Default value is sampled from (0, 1).
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - sharpness_factor (torch.Tensor): element-wise sharpness factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(sharpness, 'sharpness', bounds=(0, float('inf')))

    sharpness_factor = _adapted_uniform(
        (batch_size,),
        sharpness[0].to(device=device, dtype=dtype),
        sharpness[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    return dict(sharpness_factor=sharpness_factor.to(device=sharpness.device, dtype=sharpness.dtype))


def random_mixup_generator(
    batch_size: int,
    p: float = 0.5,
    lambda_val: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        p (flot): probability of applying mixup.
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B,).
            - mixup_lambdas (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_mixup_generator(5, 0.7)
        {'mixup_pairs': tensor([4, 0, 3, 1, 2]), 'mixup_lambdas': tensor([0.6323, 0.0000, 0.4017, 0.0223, 0.1689])}
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([lambda_val])
    lambda_val = torch.as_tensor([0.0, 1.0] if lambda_val is None else lambda_val, device=device, dtype=dtype)
    _joint_range_check(lambda_val, 'lambda_val', bounds=(0, 1))

    batch_probs: torch.Tensor = random_prob_generator(
        batch_size, p, same_on_batch=same_on_batch, device=device, dtype=dtype
    )
    mixup_pairs: torch.Tensor = torch.randperm(batch_size, device=device, dtype=dtype).long()
    mixup_lambdas: torch.Tensor = _adapted_uniform(
        (batch_size,), lambda_val[0], lambda_val[1], same_on_batch=same_on_batch
    )
    mixup_lambdas = mixup_lambdas * batch_probs

    return dict(
        mixup_pairs=mixup_pairs.to(device=_device, dtype=torch.long),
        mixup_lambdas=mixup_lambdas.to(device=_device, dtype=_dtype),
    )


def random_cutmix_generator(
    batch_size: int,
    width: int,
    height: int,
    p: float = 0.5,
    num_mix: int = 1,
    beta: Optional[torch.Tensor] = None,
    cut_size: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        width (int): image width.
        height (int): image height.
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size (torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (num_mix, B).
            - crop_src (torch.Tensor): element-wise probabilities with a shape of (num_mix, B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_cutmix_generator(3, 224, 224, p=0.5, num_mix=2)
        {'mix_pairs': tensor([[2, 0, 1],
                [1, 2, 0]]), 'crop_src': tensor([[[[ 35.,  25.],
                  [208.,  25.],
                  [208., 198.],
                  [ 35., 198.]],
        <BLANKLINE>
                 [[156., 137.],
                  [155., 137.],
                  [155., 136.],
                  [156., 136.]],
        <BLANKLINE>
                 [[  3.,  12.],
                  [210.,  12.],
                  [210., 219.],
                  [  3., 219.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 83., 125.],
                  [177., 125.],
                  [177., 219.],
                  [ 83., 219.]],
        <BLANKLINE>
                 [[ 54.,   8.],
                  [205.,   8.],
                  [205., 159.],
                  [ 54., 159.]],
        <BLANKLINE>
                 [[ 97.,  70.],
                  [ 96.,  70.],
                  [ 96.,  69.],
                  [ 97.,  69.]]]])}
    """
    _device, _dtype = _extract_device_dtype([beta, cut_size])
    beta = torch.as_tensor(1.0 if beta is None else beta, device=device, dtype=dtype)
    cut_size = torch.as_tensor([0.0, 1.0] if cut_size is None else cut_size, device=device, dtype=dtype)
    if not (num_mix >= 1 and isinstance(num_mix, (int,))):
        raise AssertionError(f"`num_mix` must be an integer greater than 1. Got {num_mix}.")
    if not (type(height) is int and height > 0 and type(width) is int and width > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
    _joint_range_check(cut_size, 'cut_size', bounds=(0, 1))
    _common_param_check(batch_size, same_on_batch)

    if batch_size == 0:
        return dict(
            mix_pairs=torch.zeros([0, 3], device=_device, dtype=torch.long),
            crop_src=torch.zeros([0, 4, 2], device=_device, dtype=torch.long),
        )

    batch_probs: torch.Tensor = random_prob_generator(
        batch_size * num_mix, p, same_on_batch, device=device, dtype=dtype
    )
    mix_pairs: torch.Tensor = torch.rand(num_mix, batch_size, device=device, dtype=dtype).argsort(dim=1)
    cutmix_betas: torch.Tensor = _adapted_beta((batch_size * num_mix,), beta, beta, same_on_batch=same_on_batch)
    # Note: torch.clamp does not accept tensor, cutmix_betas.clamp(cut_size[0], cut_size[1]) throws:
    # Argument 1 to "clamp" of "_TensorBase" has incompatible type "Tensor"; expected "float"
    cutmix_betas = torch.min(torch.max(cutmix_betas, cut_size[0]), cut_size[1])
    cutmix_rate = torch.sqrt(1.0 - cutmix_betas) * batch_probs

    cut_height = (cutmix_rate * height).floor().to(device=device, dtype=_dtype)
    cut_width = (cutmix_rate * width).floor().to(device=device, dtype=_dtype)
    _gen_shape = (1,)

    if same_on_batch:
        _gen_shape = (cut_height.size(0),)
        cut_height = cut_height[0]
        cut_width = cut_width[0]

    # Reserve at least 1 pixel for cropping.
    x_start = (
        _adapted_uniform(
            _gen_shape,
            torch.zeros_like(cut_width, device=device, dtype=dtype),
            (width - cut_width - 1).to(device=device, dtype=dtype),
            same_on_batch,
        )
        .floor()
        .to(device=device, dtype=_dtype)
    )
    y_start = (
        _adapted_uniform(
            _gen_shape,
            torch.zeros_like(cut_height, device=device, dtype=dtype),
            (height - cut_height - 1).to(device=device, dtype=dtype),
            same_on_batch,
        )
        .floor()
        .to(device=device, dtype=_dtype)
    )

    crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

    # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
    crop_src = crop_src.view(num_mix, batch_size, 4, 2)

    return dict(
        mix_pairs=mix_pairs.to(device=_device, dtype=torch.long),
        crop_src=crop_src.floor().to(device=_device, dtype=_dtype),
    )
