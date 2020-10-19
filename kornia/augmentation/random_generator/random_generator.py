from typing import Tuple, List, Union, Dict, Optional, cast
import random

import torch
from torch.distributions import Bernoulli

from kornia.constants import Resample, BorderType, SamplePadding
from kornia.geometry import bbox_generator
from ..utils import (
    _adapted_sampling,
    _adapted_uniform,
    _adapted_beta,
    _joint_range_check,
)


def random_prob_generator(
        batch_size: int, p: float = 0.5, same_on_batch: bool = False) -> torch.Tensor:
    r"""Generate random probabilities for a batch of inputs.

    Args:
        batch_size (int): the number of images.
        p (float): probability to generate an 1-d binary mask. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        torch.Tensor: parameters to be passed for transformation.
    """
    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    probs: torch.Tensor = _adapted_sampling((batch_size,), Bernoulli(p), same_on_batch).bool()

    return probs


def random_color_jitter_generator(
    batch_size: int,
    brightness: Optional[torch.Tensor] = None,
    contrast: Optional[torch.Tensor] = None,
    saturation: Optional[torch.Tensor] = None,
    hue: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate random color jiter parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        brightness (torch.Tensor, optional): Default value is 0.
        contrast (torch.Tensor, optional): Default value is 0.
        saturation (torch.Tensor, optional): Default value is 0.
        hue (torch.Tensor, optional): Default value is 0.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    brightness = torch.tensor(0.) if brightness is None else cast(torch.Tensor, brightness)
    contrast = torch.tensor(0.) if contrast is None else cast(torch.Tensor, contrast)
    hue = torch.tensor(0.) if hue is None else cast(torch.Tensor, hue)
    saturation = torch.tensor(0.) if saturation is None else cast(torch.Tensor, saturation)

    _joint_range_check(brightness, "brightness", (0, 2))
    _joint_range_check(contrast, "contrast", (0, float('inf')))
    _joint_range_check(hue, "hue", (-0.5, 0.5))
    _joint_range_check(saturation, "saturation", (0, float('inf')))

    brightness_factor = _adapted_uniform((batch_size,), brightness[0], brightness[1], same_on_batch)
    contrast_factor = _adapted_uniform((batch_size,), contrast[0], contrast[1], same_on_batch)
    hue_factor = _adapted_uniform((batch_size,), hue[0], hue[1], same_on_batch)
    saturation_factor = _adapted_uniform((batch_size,), saturation[0], saturation[1], same_on_batch)

    return dict(brightness_factor=brightness_factor,
                contrast_factor=contrast_factor,
                hue_factor=hue_factor,
                saturation_factor=saturation_factor,
                order=torch.randperm(4))


def random_perspective_generator(
    batch_size: int,
    height: int,
    width: int,
    distortion_scale: torch.Tensor,
    same_on_batch: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params (Dict[str, torch.Tensor])
    """
    assert distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1, \
        f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}"

    start_points: torch.Tensor = torch.tensor([[
        [0., 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]]).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx = distortion_scale * width / 2
    fy = distortion_scale * height / 2

    factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

    # TODO: This line somehow breaks the gradcheck
    rand_val: torch.Tensor = _adapted_uniform(start_points.shape, 0, 1, same_on_batch)

    pts_norm = torch.tensor([[
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ]])
    end_points = start_points + factor * rand_val * pts_norm

    return dict(start_points=start_points,
                end_points=end_points)


def random_affine_generator(
    batch_size: int,
    height: int,
    width: int,
    degrees: torch.Tensor,
    translate: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shear: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``affine`` for a random affine transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        degrees (float or tuple): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(degrees, "degrees")

    angle = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)

    # compute tensor ranges
    if scale is not None:
        _joint_range_check(cast(torch.Tensor, scale[:2]), "scale")
        _scale = _adapted_uniform((batch_size,), scale[0], scale[1], same_on_batch).unsqueeze(1).repeat(1, 2)
        if len(_scale) == 4:
            _joint_range_check(cast(torch.Tensor, scale[2:]), "scale_y")
            _scale[:, 1] = _adapted_uniform((batch_size,), scale[2], scale[3], same_on_batch)
    else:
        _scale = torch.ones((batch_size, 2))

    if translate is not None:
        _joint_range_check(cast(torch.Tensor, translate), "translate")
        max_dx: torch.Tensor = translate[0] * width
        max_dy: torch.Tensor = translate[1] * height
        translations = torch.stack([
            _adapted_uniform((batch_size,), -max_dx, max_dx, same_on_batch),
            _adapted_uniform((batch_size,), -max_dy, max_dy, same_on_batch)
        ], dim=-1)
    else:
        translations = torch.zeros(batch_size, 2)

    center: torch.Tensor = torch.tensor(
        [width, height], dtype=torch.float32).view(1, 2) / 2. - 0.5
    center = center.expand(batch_size, -1)

    if shear is not None:
        _joint_range_check(cast(torch.Tensor, shear)[0], "shear")
        _joint_range_check(cast(torch.Tensor, shear)[1], "shear")
        sx = _adapted_uniform((batch_size,), shear[0][0], shear[0][1], same_on_batch)
        sy = _adapted_uniform((batch_size,), shear[1][0], shear[1][1], same_on_batch)
    else:
        sx = sy = torch.tensor([0] * batch_size)

    return dict(translations=translations,
                center=center,
                scale=_scale,
                angle=angle,
                sx=sx,
                sy=sy)


def random_rotation_generator(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): range of degrees with shape (2) to select from.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(degrees, "degrees")

    degrees = _adapted_uniform((batch_size,), degrees[0], degrees[1], same_on_batch)

    return dict(degrees=degrees)


def random_crop_generator(
    batch_size: int,
    input_size: Tuple[int, int],
    size: Union[Tuple[int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int]] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (h, w).
        size (tuple): Desired size of the crop operation, like (h, w).
            If tensor, it must be (B, 2).
        resize_to (tuple): Desired output size of the crop, like (h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> crop_size = random_crop_size_generator(
        ...     3, (30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))['size']
        >>> crop_size
        tensor([[26, 29],
                [27, 28],
                [25, 28]], dtype=torch.int32)
        >>> random_crop_generator(3, (30, 30), size=crop_size, same_on_batch=False)
        {'src': tensor([[[ 1,  3],
                 [29,  3],
                 [29, 28],
                 [ 1, 28]],
        <BLANKLINE>
                [[ 2,  3],
                 [29,  3],
                 [29, 29],
                 [ 2, 29]],
        <BLANKLINE>
                [[ 0,  2],
                 [27,  2],
                 [27, 26],
                 [ 0, 26]]]), 'dst': tensor([[[ 0,  0],
                 [28,  0],
                 [28, 25],
                 [ 0, 25]],
        <BLANKLINE>
                [[ 0,  0],
                 [27,  0],
                 [27, 26],
                 [ 0, 26]],
        <BLANKLINE>
                [[ 0,  0],
                 [27,  0],
                 [27, 24],
                 [ 0, 24]]])}
    """
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size).repeat(batch_size, 1)
    assert size.shape == torch.Size([batch_size, 2]), \
        f"If `size` is a tensor, it must be shaped as (B, 2). Got {size.shape}."
    size = size.long()

    x_diff = input_size[1] - size[:, 1] + 1
    y_diff = input_size[0] - size[:, 0] + 1

    if (x_diff < 0).any() or (y_diff < 0).any():
        raise ValueError("input_size %s cannot be smaller than crop size %s in any dimension."
                         % (str(input_size), str(size)))

    if same_on_batch:
        # If same_on_batch, select the first then repeat.
        x_start = _adapted_uniform((batch_size,), 0, x_diff[0], same_on_batch).long()
        y_start = _adapted_uniform((batch_size,), 0, y_diff[0], same_on_batch).long()
    else:
        x_start = _adapted_uniform((1,), 0, x_diff, same_on_batch).long()
        y_start = _adapted_uniform((1,), 0, y_diff, same_on_batch).long()

    crop_src = bbox_generator(x_start.view(-1), y_start.view(-1), size[:, 1] - 1, size[:, 0] - 1)

    if resize_to is None:
        crop_dst = bbox_generator(
            torch.tensor([0] * batch_size), torch.tensor([0] * batch_size), size[:, 1] - 1, size[:, 0] - 1)
    else:
        crop_dst = torch.tensor([[
            [0, 0],
            [resize_to[1] - 1, 0],
            [resize_to[1] - 1, resize_to[0] - 1],
            [0, resize_to[0] - 1],
        ]]).repeat(batch_size, 1, 1)

    return dict(src=crop_src,
                dst=crop_dst)


def random_crop_size_generator(
    batch_size: int,
    size: Tuple[int, int],
    scale: torch.Tensor,
    ratio: torch.Tensor,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Get cropping heights and widths for ```crop``` transformation for resized crop transform.

    Args:
        batch_size (int): the tensor batch size.
        size (Tuple[int, int]): expected output size of each edge.
        scale (tensor): range of size of the origin size cropped with (2,) shape.
        ratio (tensor): range of aspect ratio of the origin aspect ratio cropped with (2,) shape.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> random_crop_size_generator(3, (30, 30), scale=torch.tensor([.7, 1.3]), ratio=torch.tensor([.9, 1.]))
        {'size': tensor([[26, 29],
                [27, 28],
                [25, 28]], dtype=torch.int32)}
    """
    _joint_range_check(scale, "scale")
    _joint_range_check(ratio, "ratio")

    # 10 trails for each element
    area = _adapted_uniform(
        (batch_size, 10), scale[0] * size[0] * size[1], scale[1] * size[0] * size[1], same_on_batch)
    log_ratio = _adapted_uniform(
        (batch_size, 10), torch.log(ratio[0]), torch.log(ratio[1]), same_on_batch)
    aspect_ratio = torch.exp(log_ratio)

    w = torch.sqrt(area * aspect_ratio).int()
    h = torch.sqrt(area / aspect_ratio).int()
    # Element-wise w, h condition
    cond = ((0 < h) * (h < size[1]) * (0 < w) * (w < size[0])).int()
    cond_bool = torch.sum(cond, dim=1) > 0

    w_out = w[torch.arange(0, batch_size), torch.argmax(cond, dim=1)]
    h_out = h[torch.arange(0, batch_size), torch.argmax(cond, dim=1)]

    if not cond_bool.all():
        # Fallback to center crop
        in_ratio = float(size[0]) / float(size[1])
        if (in_ratio < min(ratio)):
            w_ct = torch.tensor(size[0])
            h_ct = torch.round(w_ct / min(ratio))
        elif (in_ratio > max(ratio)):
            h_ct = torch.tensor(size[1])
            w_ct = torch.round(h_ct * max(ratio))
        else:  # whole image
            w_ct = torch.tensor(size[0])
            h_ct = torch.tensor(size[1])
        w_ct = w_ct.int()
        h_ct = h_ct.int()

        w_out = w_out.where(cond_bool, w_ct)
        h_out = h_out.where(cond_bool, h_ct)

    return dict(size=torch.stack([w_out, h_out], dim=1))


def random_rectangles_params_generator(
    batch_size: int,
    height: int,
    width: int,
    scale: torch.Tensor,
    ratio: torch.Tensor,
    value: float = 0.,
    same_on_batch: bool = False
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

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(scale, 'scale', bounds=(0, float('inf')))
    _joint_range_check(ratio, 'ratio', bounds=(0, float('inf')))

    zeros = torch.zeros((batch_size,))
    images_area = height * width
    target_areas = _adapted_uniform(
        (batch_size,), scale[0], scale[1], same_on_batch) * images_area
    if ratio[0] < 1. and ratio[1] > 1.:
        aspect_ratios1 = _adapted_uniform((batch_size,), ratio[0], 1, same_on_batch)
        aspect_ratios2 = _adapted_uniform((batch_size,), 1, ratio[1], same_on_batch)
        rand_idxs = torch.round(torch.rand((batch_size,))).bool()
        aspect_ratios = torch.where(rand_idxs, aspect_ratios1, aspect_ratios2)
    else:
        aspect_ratios = _adapted_uniform((batch_size,), ratio[0], ratio[1], same_on_batch)

    # based on target areas and aspect ratios, rectangle params are computed
    heights = torch.min(
        torch.max(torch.round((target_areas * aspect_ratios) ** (1 / 2)),
                  torch.tensor(1.)),
        torch.tensor(float(height))
    ).int()

    widths = torch.min(
        torch.max(torch.round((target_areas / aspect_ratios) ** (1 / 2)),
                  torch.tensor(1.)),
        torch.tensor(float(width))
    ).int()

    xs = (_adapted_uniform((batch_size,), 0, 1, same_on_batch) * (torch.tensor(width) - widths + 1).float()).int()
    ys = (_adapted_uniform((batch_size,), 0, 1, same_on_batch) * (torch.tensor(height) - heights + 1).float()).int()

    return dict(widths=widths,
                heights=heights,
                xs=xs,
                ys=ys,
                values=torch.tensor([value] * batch_size))


def center_crop_generator(
    batch_size: int,
    height: int,
    width: int,
    size: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```center_crop``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (h, w).

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))

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
    points_src: torch.Tensor = torch.tensor([[
        [start_x, start_y],
        [end_x, start_y],
        [end_x, end_y],
        [start_x, end_y],
    ]])

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]]).expand(points_src.shape[0], -1, -1)
    return dict(src=points_src,
                dst=points_dst)


def random_motion_blur_generator(
    batch_size: int,
    kernel_size: Union[int, Tuple[int, int]],
    angle: torch.Tensor,
    direction: torch.Tensor,
    same_on_batch: bool = False
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

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(angle, 'angle')
    _joint_range_check(direction, 'direction')

    if isinstance(kernel_size, int):
        ksize_factor = torch.tensor([kernel_size] * batch_size)
    elif isinstance(kernel_size, tuple):
        # kernel_size is fixed across the batch
        ksize_factor = _adapted_uniform(
            (batch_size,), kernel_size[0] // 2, kernel_size[1] // 2, same_on_batch=True).int() * 2 + 1
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    angle_factor = _adapted_uniform(
        (batch_size,), angle[0], angle[1], same_on_batch)

    direction_factor = _adapted_uniform(
        (batch_size,), direction[0], direction[1], same_on_batch)

    return dict(ksize_factor=ksize_factor,
                angle_factor=angle_factor,
                direction_factor=direction_factor)


def random_solarize_generator(
    batch_size: int,
    thresholds: torch.Tensor = torch.tensor([0.4, 0.6]),
    additions: torch.Tensor = torch.tensor([-0.1, 0.1]),
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate random solarize parameters for a batch of images.

    For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the pixel value
    to be between 0 and 1.0

    Args:
        batch_size (int): the number of images.
        thresholds (torch.Tensor): Pixels less than threshold will selected. Otherwise, subtract 1.0 from the pixel.
            Default value will be sampled from [0.4, 0.6].
        additions (torch.Tensor): The value is between -0.5 and 0.5. Default value will be sampled from [-0.1, 0.1]
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(thresholds, 'thresholds')
    _joint_range_check(additions, 'additions')

    thresholds_factor = _adapted_uniform(
        (batch_size,), thresholds[0], thresholds[1], same_on_batch)

    additions_factor = _adapted_uniform(
        (batch_size,), additions[0], additions[1], same_on_batch)

    return dict(
        thresholds_factor=thresholds_factor,
        additions_factor=additions_factor
    )


def random_posterize_generator(
    batch_size: int,
    bits: torch.Tensor = torch.tensor(3),
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate random posterize parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        bits (int or tuple): Default value is 3. Integer that ranged from 0 ~ 8.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    bits_factor = _adapted_uniform((batch_size,), bits[0], bits[1], same_on_batch).int()

    return dict(
        bits_factor=bits_factor
    )


def random_sharpness_generator(
    batch_size: int,
    sharpness: torch.Tensor = torch.tensor([0, 1.]),
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate random sharpness parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        sharpness (torch.Tensor): Must be above 0. Default value is sampled from (0, 1).
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
    """
    _joint_range_check(sharpness, 'sharpness', bounds=(0, float('inf')))

    sharpness_factor = _adapted_uniform((batch_size,), sharpness[0], sharpness[1], same_on_batch)

    return dict(
        sharpness_factor=sharpness_factor
    )


def random_mixup_generator(
    batch_size: int,
    p: float = 0.5,
    lambda_val: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        p (flot): probability of applying mixup.
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_mixup_generator(5, 0.7)
        {'mixup_pairs': tensor([4, 0, 3, 1, 2]), 'mixup_lambdas': tensor([0.6323, 0.0000, 0.4017, 0.0223, 0.1689])}
    """
    if lambda_val is None:
        lambda_val = torch.tensor([0., 1.])
    _joint_range_check(lambda_val, 'lambda_val', bounds=(0, 1))

    batch_probs: torch.Tensor = random_prob_generator(batch_size, p, same_on_batch=same_on_batch)
    mixup_pairs: torch.Tensor = torch.randperm(batch_size)
    mixup_lambdas: torch.Tensor = _adapted_uniform(
        (batch_size,), lambda_val[0], lambda_val[1], same_on_batch=same_on_batch)
    mixup_lambdas = mixup_lambdas * batch_probs.float()

    return dict(
        mixup_pairs=mixup_pairs,
        mixup_lambdas=mixup_lambdas
    )


def random_cutmix_generator(
    batch_size: int,
    width: int,
    height: int,
    p: float = 0.5,
    num_mix: int = 1,
    beta: Optional[torch.Tensor] = None,
    cut_size: Optional[torch.Tensor] = None,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        width (int): image width.
        height (int): image height.
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (float or torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size ((float, float) or torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_cutmix_generator(3, 224, 224, p=0.5, num_mix=2)
        {'mix_pairs': tensor([[2, 0, 1],
                [1, 2, 0]]), 'crop_src': tensor([[[[ 36,  25],
                  [209,  25],
                  [209, 198],
                  [ 36, 198]],
        <BLANKLINE>
                 [[157, 137],
                  [156, 137],
                  [156, 136],
                  [157, 136]],
        <BLANKLINE>
                 [[  3,  12],
                  [210,  12],
                  [210, 219],
                  [  3, 219]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 83, 126],
                  [177, 126],
                  [177, 220],
                  [ 83, 220]],
        <BLANKLINE>
                 [[ 55,   8],
                  [206,   8],
                  [206, 159],
                  [ 55, 159]],
        <BLANKLINE>
                 [[ 97,  70],
                  [ 96,  70],
                  [ 96,  69],
                  [ 97,  69]]]])}

    """
    if beta is None:
        beta = torch.tensor(1.)
    if cut_size is None:
        cut_size = torch.tensor([0., 1.])
    assert num_mix >= 1 and isinstance(num_mix, (int,)), \
        f"`num_mix` must be an integer greater than 1. Got {num_mix}."
    _joint_range_check(cut_size, 'cut_size', bounds=(0, 1))

    batch_probs: torch.Tensor = random_prob_generator(batch_size * num_mix, p, same_on_batch)
    mix_pairs: torch.Tensor = torch.rand(num_mix, batch_size).argsort(dim=1)
    cutmix_betas: torch.Tensor = _adapted_beta((batch_size * num_mix,), beta, beta, same_on_batch=same_on_batch)
    # Note: torch.clamp does not accept tensor, cutmix_betas.clamp(cut_size[0], cut_size[1]) throws:
    # Argument 1 to "clamp" of "_TensorBase" has incompatible type "Tensor"; expected "float"
    cutmix_betas = torch.min(torch.max(cutmix_betas, cut_size[0]), cut_size[1])
    cutmix_rate = torch.sqrt(1. - cutmix_betas) * batch_probs

    cut_height = (cutmix_rate * height).long() - 1
    cut_width = (cutmix_rate * width).long() - 1
    _gen_shape = (1,)

    if same_on_batch:
        _gen_shape = (cut_height.size(0),)
        cut_height = cut_height[0]
        cut_width = cut_width[0]

    # Reserve at least 1 pixel for cropping.
    x_start = _adapted_uniform(
        _gen_shape, torch.zeros_like(cut_width, dtype=torch.float32), width - cut_width - 1, same_on_batch).long()
    y_start = _adapted_uniform(
        _gen_shape, torch.zeros_like(cut_height, dtype=torch.float32), height - cut_height - 1, same_on_batch).long()

    crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

    # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
    crop_src = crop_src.view(num_mix, batch_size, 4, 2)

    return dict(
        mix_pairs=mix_pairs,
        crop_src=crop_src
    )
