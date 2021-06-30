from typing import cast, Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli

from kornia.geometry.bbox import bbox_generator
from kornia.utils import _extract_device_dtype

from ..utils import _adapted_beta, _adapted_sampling, _adapted_uniform, _common_param_check, _joint_range_check


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
    assert (
        distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1
    ), f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}."
    assert (
        type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'height' and 'width' must be integers. Got {height}, {width}."

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
    assert (
        isinstance(width, (int,)) and isinstance(height, (int,)) and width > 0 and height > 0
    ), f"`width` and `height` must be positive integers. Got {width}, {height}."

    _device, _dtype = _extract_device_dtype([degrees, translate, scale, shear])
    degrees = degrees.to(device=device, dtype=dtype)
    angle = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)
    angle = angle.to(device=_device, dtype=_dtype)

    # compute tensor ranges
    if scale is not None:
        scale = scale.to(device=device, dtype=dtype)
        assert len(scale.shape) == 1 and (
            len(scale) == 2 or len(scale) == 4
        ), f"`scale` shall have 2 or 4 elements. Got {scale}."
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
        assert (
            0.0 <= translate[0] <= 1.0 and 0.0 <= translate[1] <= 1.0 and translate.shape == torch.Size([2])
        ), f"Expect translate contains two elements and ranges are in [0, 1]. Got {translate}."
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
    assert size.shape == torch.Size([batch_size, 2]), (
        "If `size` is a tensor, it must be shaped as (B, 2). "
        f"Got {size.shape} while expecting {torch.Size([batch_size, 2])}."
    )
    assert (
        input_size[0] > 0 and input_size[1] > 0 and (size > 0).all()
    ), f"Got non-positive input size or size. {input_size}, {size}."
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
        assert (
            len(resize_to) == 2
            and isinstance(resize_to[0], (int,))
            and isinstance(resize_to[1], (int,))
            and resize_to[0] > 0
            and resize_to[1] > 0
        ), f"`resize_to` must be a tuple of 2 positive integers. Got {resize_to}."
        crop_dst = torch.tensor(
            [[[0, 0], [resize_to[1] - 1, 0], [resize_to[1] - 1, resize_to[0] - 1], [0, resize_to[0] - 1]]],
            device=_device,
            dtype=_dtype,
        ).repeat(batch_size, 1, 1)

    _input_size = torch.tensor(input_size, device=_device, dtype=torch.long).expand(batch_size, -1)

    return dict(src=crop_src, dst=crop_dst, input_size=_input_size)


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
    assert (
        len(size) == 2 and type(size[0]) is int and size[1] > 0 and type(size[1]) is int and size[1] > 0
    ), f"'height' and 'width' must be integers. Got {size}."

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

    # torch.argmax is not reproducible accross devices: https://github.com/pytorch/pytorch/issues/17738
    # Here, we will select the first occurance of the duplicated elements.
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
    assert (
        type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'height' and 'width' must be integers. Got {height}, {width}."
    assert (
        isinstance(value, (int, float)) and value >= 0 and value <= 1
    ), f"'value' must be a number between 0 - 1. Got {value}."
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
        raise ValueError("Input size must be a tuple/list of length 2. Got {}".format(size))
    assert (
        type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'height' and 'width' must be integers. Got {height}, {width}."
    assert (
        height >= size[0] and width >= size[1]
    ), f"Crop size must be smaller than input size. Got ({height}, {width}) and {size}."

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = height, width

    # compute start/end offsets
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half

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
        assert (
            kernel_size >= 3 and kernel_size % 2 == 1
        ), f"`kernel_size` must be odd and greater than 3. Got {kernel_size}."
        ksize_factor = torch.tensor([kernel_size] * batch_size, device=device, dtype=dtype)
    elif isinstance(kernel_size, tuple):
        # kernel_size is fixed across the batch
        assert len(kernel_size) == 2, f"`kernel_size` must be (2,) if it is a tuple. Got {kernel_size}."
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
    assert num_mix >= 1 and isinstance(num_mix, (int,)), f"`num_mix` must be an integer greater than 1. Got {num_mix}."
    assert (
        type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'height' and 'width' must be integers. Got {height}, {width}."
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
