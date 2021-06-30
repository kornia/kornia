from typing import Dict, Optional, Tuple, Union

import torch

from kornia.geometry.bbox import bbox_generator3d
from kornia.utils import _extract_device_dtype

from ..utils import _adapted_uniform, _joint_range_check


def random_rotation_generator3d(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): Ranges of degrees (3, 2) for yaw, pitch and roll.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - yaw (torch.Tensor): element-wise rotation yaws with a shape of (B,).
            - pitch (torch.Tensor): element-wise rotation pitches with a shape of (B,).
            - roll (torch.Tensor): element-wise rotation rolls with a shape of (B,).
    """
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    _device, _dtype = _extract_device_dtype([degrees])
    degrees = degrees.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)

    return dict(
        yaw=yaw.to(device=_device, dtype=_dtype),
        pitch=pitch.to(device=_device, dtype=_dtype),
        roll=roll.to(device=_device, dtype=_dtype),
    )


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
    assert (
        type(depth) is int and depth > 0 and type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'depth', 'height' and 'width' must be integers. Got {depth}, {height}, {width}."

    _device, _dtype = _extract_device_dtype([degrees, translate, scale, shears])
    assert degrees.shape == torch.Size([3, 2]), f"'degrees' must be the shape of (3, 2). Got {degrees.shape}."
    degrees = degrees.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), degrees[0][0], degrees[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), degrees[1][0], degrees[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), degrees[2][0], degrees[2][1], same_on_batch)
    angles = torch.stack([yaw, pitch, roll], dim=1)

    # compute tensor ranges
    if scale is not None:
        assert scale.shape == torch.Size([3, 2]), f"'scale' must be the shape of (3, 2). Got {scale.shape}."
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
        scale = torch.ones(batch_size, device=device, dtype=dtype).repeat(1, 3)

    if translate is not None:
        assert translate.shape == torch.Size([3]), f"'translate' must be the shape of (2). Got {translate.shape}."
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
        assert shears.shape == torch.Size([6, 2]), f"'shears' must be the shape of (6, 2). Got {shears.shape}."
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


def random_motion_blur_generator3d(
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
        angle (torch.Tensor): yaw, pitch and roll range of the motion blur in degrees :math:`(3, 2)`.
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
            - angle_factor (torch.Tensor): element-wise center with a shape of (B,).
            - direction_factor (torch.Tensor): element-wise scales with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _device, _dtype = _extract_device_dtype([angle, direction])
    _joint_range_check(direction, 'direction', (-1, 1))
    if isinstance(kernel_size, int):
        assert (
            kernel_size >= 3 and kernel_size % 2 == 1
        ), f"`kernel_size` must be odd and greater than 3. Got {kernel_size}."
        ksize_factor = torch.tensor([kernel_size] * batch_size, device=device, dtype=dtype).int()
    elif isinstance(kernel_size, tuple):
        assert (
            len(kernel_size) == 2 and kernel_size[0] >= 3 and kernel_size[0] <= kernel_size[1]
        ), f"`kernel_size` must be greater than 3. Got range {kernel_size}."
        # kernel_size is fixed across the batch
        ksize_factor = (
            _adapted_uniform((batch_size,), kernel_size[0] // 2, kernel_size[1] // 2, same_on_batch=True).int() * 2 + 1
        )
    else:
        raise TypeError(f"Unsupported type: {type(kernel_size)}")

    assert angle.shape == torch.Size([3, 2]), f"'angle' must be the shape of (3, 2). Got {angle.shape}."
    angle = angle.to(device=device, dtype=dtype)
    yaw = _adapted_uniform((batch_size,), angle[0][0], angle[0][1], same_on_batch)
    pitch = _adapted_uniform((batch_size,), angle[1][0], angle[1][1], same_on_batch)
    roll = _adapted_uniform((batch_size,), angle[2][0], angle[2][1], same_on_batch)
    angle_factor = torch.stack([yaw, pitch, roll], dim=1)

    direction = direction.to(device=device, dtype=dtype)
    direction_factor = _adapted_uniform((batch_size,), direction[0], direction[1], same_on_batch)

    return dict(
        ksize_factor=ksize_factor.to(device=_device),
        angle_factor=angle_factor.to(device=_device, dtype=_dtype),
        direction_factor=direction_factor.to(device=_device, dtype=_dtype),
    )


def center_crop_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    size: Tuple[int, int, int],
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```center_crop3d``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (d, h, w).
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        No random number will be generated.
    """
    if not isinstance(size, (tuple, list)) and len(size) == 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}".format(size))
    assert (
        type(depth) is int and depth > 0 and type(height) is int and height > 0 and type(width) is int and width > 0
    ), f"'depth', 'height' and 'width' must be integers. Got {depth}, {height}, {width}."
    assert (
        depth >= size[0] and height >= size[1] and width >= size[2]
    ), f"Crop size must be smaller than input size. Got ({depth}, {height}, {width}) and {size}."

    if batch_size == 0:
        return dict(src=torch.zeros([0, 8, 3]), dst=torch.zeros([0, 8, 3]))
    # unpack input sizes
    dst_d, dst_h, dst_w = size
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
    points_src: torch.Tensor = torch.tensor(
        [
            [
                [start_x, start_y, start_z],
                [end_x, start_y, start_z],
                [end_x, end_y, start_z],
                [start_x, end_y, start_z],
                [start_x, start_y, end_z],
                [end_x, start_y, end_z],
                [end_x, end_y, end_z],
                [start_x, end_y, end_z],
            ]
        ],
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1, -1)

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: torch.Tensor = torch.tensor(
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
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1, -1)
    return dict(src=points_src, dst=points_dst)


def random_crop_generator3d(
    batch_size: int,
    input_size: Tuple[int, int, int],
    size: Union[Tuple[int, int, int], torch.Tensor],
    resize_to: Optional[Tuple[int, int, int]] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        batch_size (int): the tensor batch size.
        input_size (tuple): Input image shape, like (d, h, w).
        size (tuple): Desired size of the crop operation, like (d, h, w).
            If tensor, it must be (B, 3).
        resize_to (tuple): Desired output size of the crop, like (d, h, w). If None, no resize will be performed.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (torch.Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _device, _dtype = _extract_device_dtype([size if isinstance(size, torch.Tensor) else None])
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=device, dtype=dtype).repeat(batch_size, 1)
    else:
        size = size.to(device=device, dtype=dtype)
    assert size.shape == torch.Size([batch_size, 3]), (
        "If `size` is a tensor, it must be shaped as (B, 3). "
        f"Got {size.shape} while expecting {torch.Size([batch_size, 3])}."
    )
    assert (
        len(input_size) == 3
        and isinstance(input_size[0], (int,))
        and isinstance(input_size[1], (int,))
        and isinstance(input_size[2], (int,))
        and input_size[0] > 0
        and input_size[1] > 0
        and input_size[2] > 0
    ), f"`input_size` must be a tuple of 3 positive integers. Got {input_size}."

    x_diff = input_size[2] - size[:, 2] + 1
    y_diff = input_size[1] - size[:, 1] + 1
    z_diff = input_size[0] - size[:, 0] + 1

    if (x_diff < 0).any() or (y_diff < 0).any() or (z_diff < 0).any():
        raise ValueError(
            "input_size %s cannot be smaller than crop size %s in any dimension." % (str(input_size), str(size))
        )

    if batch_size == 0:
        return dict(
            src=torch.zeros([0, 8, 3], device=_device, dtype=_dtype),
            dst=torch.zeros([0, 8, 3], device=_device, dtype=_dtype),
        )

    if same_on_batch:
        # If same_on_batch, select the first then repeat.
        x_start = _adapted_uniform((batch_size,), 0, x_diff[0], same_on_batch).floor()
        y_start = _adapted_uniform((batch_size,), 0, y_diff[0], same_on_batch).floor()
        z_start = _adapted_uniform((batch_size,), 0, z_diff[0], same_on_batch).floor()
    else:
        x_start = _adapted_uniform((1,), 0, x_diff, same_on_batch).floor()
        y_start = _adapted_uniform((1,), 0, y_diff, same_on_batch).floor()
        z_start = _adapted_uniform((1,), 0, z_diff, same_on_batch).floor()

    crop_src = bbox_generator3d(
        x_start.to(device=_device, dtype=_dtype).view(-1),
        y_start.to(device=_device, dtype=_dtype).view(-1),
        z_start.to(device=_device, dtype=_dtype).view(-1),
        size[:, 2].to(device=_device, dtype=_dtype) - 1,
        size[:, 1].to(device=_device, dtype=_dtype) - 1,
        size[:, 0].to(device=_device, dtype=_dtype) - 1,
    )

    if resize_to is None:
        crop_dst = bbox_generator3d(
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            torch.tensor([0] * batch_size, device=_device, dtype=_dtype),
            size[:, 2].to(device=_device, dtype=_dtype) - 1,
            size[:, 1].to(device=_device, dtype=_dtype) - 1,
            size[:, 0].to(device=_device, dtype=_dtype) - 1,
        )
    else:
        assert (
            len(resize_to) == 3
            and isinstance(resize_to[0], (int,))
            and isinstance(resize_to[1], (int,))
            and isinstance(resize_to[2], (int,))
            and resize_to[0] > 0
            and resize_to[1] > 0
            and resize_to[2] > 0
        ), f"`resize_to` must be a tuple of 3 positive integers. Got {resize_to}."
        crop_dst = torch.tensor(
            [
                [
                    [0, 0, 0],
                    [resize_to[-1] - 1, 0, 0],
                    [resize_to[-1] - 1, resize_to[-2] - 1, 0],
                    [0, resize_to[-2] - 1, 0],
                    [0, 0, resize_to[-3] - 1],
                    [resize_to[-1] - 1, 0, resize_to[-3] - 1],
                    [resize_to[-1] - 1, resize_to[-2] - 1, resize_to[-3] - 1],
                    [0, resize_to[-2] - 1, resize_to[-3] - 1],
                ]
            ],
            device=_device,
            dtype=_dtype,
        ).repeat(batch_size, 1, 1)

    return dict(src=crop_src.to(device=_device), dst=crop_dst.to(device=_device))


def random_perspective_generator3d(
    batch_size: int,
    depth: int,
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
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        distortion_scale (torch.Tensor): it controls the degree of distortion and ranges from 0 to 1.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - src (torch.Tensor): perspective source bounding boxes with a shape of (B, 8, 3).
            - dst (torch.Tensor): perspective target bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    assert (
        distortion_scale.dim() == 0 and 0 <= distortion_scale <= 1
    ), f"'distortion_scale' must be a scalar within [0, 1]. Got {distortion_scale}"
    _device, _dtype = _extract_device_dtype([distortion_scale])
    distortion_scale = distortion_scale.to(device=device, dtype=dtype)

    start_points: torch.Tensor = torch.tensor(
        [
            [
                [0.0, 0, 0],
                [width - 1, 0, 0],
                [width - 1, height - 1, 0],
                [0, height - 1, 0],
                [0.0, 0, depth - 1],
                [width - 1, 0, depth - 1],
                [width - 1, height - 1, depth - 1],
                [0, height - 1, depth - 1],
            ]
        ],
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1, -1)

    # generate random offset not larger than half of the image
    fx = distortion_scale * width / 2
    fy = distortion_scale * height / 2
    fz = distortion_scale * depth / 2

    factor = torch.stack([fx, fy, fz], dim=0).view(-1, 1, 3)

    rand_val: torch.Tensor = _adapted_uniform(
        start_points.shape,
        torch.tensor(0, device=device, dtype=dtype),
        torch.tensor(1, device=device, dtype=dtype),
        same_on_batch,
    )

    pts_norm = torch.tensor(
        [[[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]],
        device=device,
        dtype=dtype,
    )
    end_points = start_points + factor * rand_val * pts_norm

    return dict(
        start_points=start_points.to(device=_device, dtype=_dtype),
        end_points=end_points.to(device=_device, dtype=_dtype),
    )
