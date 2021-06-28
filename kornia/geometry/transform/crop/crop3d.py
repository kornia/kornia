import warnings
from typing import Optional, Tuple

import torch

from kornia.geometry.bbox import bbox_generator3d as _bbox_generator3d
from kornia.geometry.bbox import bbox_to_mask3d as _bbox_to_mask3d
from kornia.geometry.bbox import infer_bbox_shape3d as _infer_bbox_shape3d
from kornia.geometry.bbox import validate_bbox3d as _validate_bbox3d
from kornia.geometry.transform.projwarp import get_perspective_transform3d, warp_affine3d

__all__ = [
    "crop_and_resize3d",
    "crop_by_boxes3d",
    "crop_by_transform_mat3d",
    "center_crop3d",
    "validate_bboxes3d",
    "infer_box_shape3d",
    "bbox_to_mask3d",
    "bbox_generator3d",
]


def crop_and_resize3d(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int, int],
    interpolation: str = 'bilinear',
    align_corners: bool = False,
) -> torch.Tensor:
    r"""Extract crops from 3D volumes (5D tensor) and resize them.

    Args:
        tensor: the 3D volume tensor with shape (B, C, D, H, W).
        boxes: a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        size: a tuple with the height and width that will be
            used to resize the extracted patches.
        interpolation: Interpolation flag.
        align_corners: mode for grid_generation.

    Returns:
        tensor containing the patches with shape (Bx)CxN1xN2xN3.

    Example:
        >>> input = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
        >>> input
        tensor([[[[[ 0.,  1.,  2.,  3.],
                   [ 4.,  5.,  6.,  7.],
                   [ 8.,  9., 10., 11.],
                   [12., 13., 14., 15.]],
        <BLANKLINE>
                  [[16., 17., 18., 19.],
                   [20., 21., 22., 23.],
                   [24., 25., 26., 27.],
                   [28., 29., 30., 31.]],
        <BLANKLINE>
                  [[32., 33., 34., 35.],
                   [36., 37., 38., 39.],
                   [40., 41., 42., 43.],
                   [44., 45., 46., 47.]],
        <BLANKLINE>
                  [[48., 49., 50., 51.],
                   [52., 53., 54., 55.],
                   [56., 57., 58., 59.],
                   [60., 61., 62., 63.]]]]])
        >>> boxes = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [3., 1., 1.],
        ...     [3., 3., 1.],
        ...     [1., 3., 1.],
        ...     [1., 1., 2.],
        ...     [3., 1., 2.],
        ...     [3., 3., 2.],
        ...     [1., 3., 2.],
        ... ]])  # 1x8x3
        >>> crop_and_resize3d(input, boxes, (2, 2, 2), align_corners=True)
        tensor([[[[[21.0000, 23.0000],
                   [29.0000, 31.0000]],
        <BLANKLINE>
                  [[37.0000, 39.0000],
                   [45.0000, 47.0000]]]]])
    """
    if not isinstance(tensor, (torch.Tensor)):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))
    if not isinstance(boxes, (torch.Tensor)):
        raise TypeError("Input boxes type is not a torch.Tensor. Got {}".format(type(boxes)))
    if not isinstance(size, (tuple, list)) and len(size) != 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}".format(size))
    assert len(tensor.shape) == 5, f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}."
    # unpack input data
    dst_d, dst_h, dst_w = size[0], size[1], size[2]

    # [x, y, z] origin
    # from front to back
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = boxes

    # [x, y, z] destination
    # from front to back
    # top-left, top-right, bottom-right, bottom-left
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
        dtype=tensor.dtype,
        device=tensor.device,
    ).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes3d(tensor, points_src, points_dst, interpolation, align_corners)


def center_crop3d(
    tensor: torch.Tensor, size: Tuple[int, int, int], interpolation: str = 'bilinear', align_corners: bool = True
) -> torch.Tensor:
    r"""Crop the 3D volumes (5D tensor) at the center.

    Args:
        tensor: the 3D volume tensor with shape (B, C, D, H, W).
        size: a tuple with the expected depth, height and width
            of the output patch.
        interpolation: Interpolation flag.
        align_corners : mode for grid_generation.

    Returns:
        the output tensor with patches.

    Examples:
        >>> input = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4)
        >>> input
        tensor([[[[[ 0.,  1.,  2.,  3.],
                   [ 4.,  5.,  6.,  7.],
                   [ 8.,  9., 10., 11.],
                   [12., 13., 14., 15.]],
        <BLANKLINE>
                  [[16., 17., 18., 19.],
                   [20., 21., 22., 23.],
                   [24., 25., 26., 27.],
                   [28., 29., 30., 31.]],
        <BLANKLINE>
                  [[32., 33., 34., 35.],
                   [36., 37., 38., 39.],
                   [40., 41., 42., 43.],
                   [44., 45., 46., 47.]],
        <BLANKLINE>
                  [[48., 49., 50., 51.],
                   [52., 53., 54., 55.],
                   [56., 57., 58., 59.],
                   [60., 61., 62., 63.]]]]])
        >>> center_crop3d(input, (2, 2, 2), align_corners=True)
        tensor([[[[[21.0000, 22.0000],
                   [25.0000, 26.0000]],
        <BLANKLINE>
                  [[37.0000, 38.0000],
                   [41.0000, 42.0000]]]]])
    """
    if not isinstance(tensor, (torch.Tensor)):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    assert len(tensor.shape) == 5, f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}."

    if not isinstance(size, (tuple, list)) and len(size) == 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}".format(size))

    # unpack input sizes
    dst_d, dst_h, dst_w = size
    src_d, src_h, src_w = tensor.shape[-3:]

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
        device=tensor.device,
    )

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
        device=tensor.device,
    ).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes3d(
        tensor, points_src.to(tensor.dtype), points_dst.to(tensor.dtype), interpolation, align_corners
    )


def crop_by_boxes3d(
    tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    interpolation: str = 'bilinear',
    align_corners: bool = False,
) -> torch.Tensor:
    """Perform crop transform on 3D volumes (5D tensor) by bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width, height and depth.

    Args:
        tensor : the 3D volume tensor with shape (B, C, D, H, W).
        src_box : a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        dst_box: a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        interpolation: Interpolation flag.
        align_corners: mode for grid_generation.

    Returns:
        the output tensor with patches.

    Examples:
        >>> input = torch.tensor([[[
        ...         [[ 0.,  1.,  2.,  3.],
        ...          [ 4.,  5.,  6.,  7.],
        ...          [ 8.,  9., 10., 11.],
        ...          [12., 13., 14., 15.]],
        ...         [[16., 17., 18., 19.],
        ...          [20., 21., 22., 23.],
        ...          [24., 25., 26., 27.],
        ...          [28., 29., 30., 31.]],
        ...         [[32., 33., 34., 35.],
        ...          [36., 37., 38., 39.],
        ...          [40., 41., 42., 43.],
        ...          [44., 45., 46., 47.]]]]])
        >>> src_box = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [3., 1., 1.],
        ...     [3., 3., 1.],
        ...     [1., 3., 1.],
        ...     [1., 1., 2.],
        ...     [3., 1., 2.],
        ...     [3., 3., 2.],
        ...     [1., 3., 2.],
        ... ]])  # 1x8x3
        >>> dst_box = torch.tensor([[
        ...     [0., 0., 0.],
        ...     [2., 0., 0.],
        ...     [2., 2., 0.],
        ...     [0., 2., 0.],
        ...     [0., 0., 1.],
        ...     [2., 0., 1.],
        ...     [2., 2., 1.],
        ...     [0., 2., 1.],
        ... ]])  # 1x8x3
        >>> crop_by_boxes3d(input, src_box, dst_box, interpolation='nearest', align_corners=True)
        tensor([[[[[21., 22., 23.],
                   [25., 26., 27.],
                   [29., 30., 31.]],
        <BLANKLINE>
                  [[37., 38., 39.],
                   [41., 42., 43.],
                   [45., 46., 47.]]]]])

    """
    _validate_bbox3d(src_box)
    _validate_bbox3d(dst_box)

    assert len(tensor.shape) == 5, f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: torch.Tensor = get_perspective_transform3d(src_box.to(tensor.dtype), dst_box.to(tensor.dtype))
    # simulate broadcasting
    dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1).type_as(tensor)

    bbox = _infer_bbox_shape3d(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all() and (bbox[2] == bbox[2][0]).all(), (
        "Cropping height, width and depth must be exact same in a batch."
        f"Got height {bbox[0]}, width {bbox[1]} and depth {bbox[2]}."
    )

    patches: torch.Tensor = crop_by_transform_mat3d(
        tensor,
        dst_trans_src,
        (int(bbox[0][0].item()), int(bbox[1][0].item()), int(bbox[2][0].item())),
        mode=interpolation,
        align_corners=align_corners,
    )

    return patches


def crop_by_transform_mat3d(
    tensor: torch.Tensor,
    transform: torch.Tensor,
    out_size: Tuple[int, int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """Perform crop transform on 3D volumes (5D tensor) given a perspective transformation matrix.

    Args:
        tensor: the 2D image tensor with shape (B, C, H, W).
        transform: a perspective transformation matrix with shape (B, 4, 4).
        out_size: size of the output image (depth, height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the output tensor with patches.
    """
    # simulate broadcasting
    dst_trans_src = transform.expand(tensor.shape[0], -1, -1)

    patches: torch.Tensor = warp_affine3d(
        tensor, dst_trans_src[:, :3, :], out_size, flags=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    return patches


@torch.jit.ignore
def validate_bboxes3d(boxes: torch.Tensor) -> bool:
    """Validate if a 3D bounding box usable or not.
    This function checks if the boxes are cube or not.
    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx8x3, where each box is defined in the following (clockwise)
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop3d.validate_bboxes3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.validate_bbox3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _validate_bbox3d(boxes)


def infer_box_shape3d(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 3D bounding boxes.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx8x3, where each box is defined in the following (clockwise)
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - Bounding box depths, shape of :math:`(B,)`.
        - Bounding box heights, shape of :math:`(B,)`.
        - Bounding box widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = torch.tensor([[[ 0,  1,  2],
        ...         [10,  1,  2],
        ...         [10, 21,  2],
        ...         [ 0, 21,  2],
        ...         [ 0,  1, 32],
        ...         [10,  1, 32],
        ...         [10, 21, 32],
        ...         [ 0, 21, 32]],
        ...        [[ 3,  4,  5],
        ...         [43,  4,  5],
        ...         [43, 54,  5],
        ...         [ 3, 54,  5],
        ...         [ 3,  4, 65],
        ...         [43,  4, 65],
        ...         [43, 54, 65],
        ...         [ 3, 54, 65]]]) # 2x8x3
        >>> infer_box_shape3d(boxes)
        (tensor([31, 61]), tensor([21, 51]), tensor([11, 41]))
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.infer_box_shape3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.infer_bbox_shape3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _infer_bbox_shape3d(boxes)


def bbox_to_mask3d(boxes: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """Convert 3D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx8x3, where each box is defined in the following (clockwise)
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.
        size (Tuple[int, int, int]): depth, height and width of the masked image.

    Returns:
        torch.Tensor: the output mask tensor.

    Examples:
        >>> boxes = torch.tensor([[
        ...     [1., 1., 1.],
        ...     [2., 1., 1.],
        ...     [2., 2., 1.],
        ...     [1., 2., 1.],
        ...     [1., 1., 2.],
        ...     [2., 1., 2.],
        ...     [2., 2., 2.],
        ...     [1., 2., 2.],
        ... ]])  # 1x8x3
        >>> bbox_to_mask3d(boxes, (4, 5, 5))
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 1., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.bbox_to_mask3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.bbox_to_mask3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _bbox_to_mask3d(boxes, size)


def bbox_generator3d(
    x_start: torch.Tensor,
    y_start: torch.Tensor,
    z_start: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Generate 3D bounding boxes according to the provided start coords, width, height and depth.

    Args:
        x_start (torch.Tensor): a tensor containing the x coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        y_start (torch.Tensor): a tensor containing the y coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        z_start (torch.Tensor): a tensor containing the z coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        width (torch.Tensor): widths of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.
        height (torch.Tensor): heights of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.
        depth (torch.Tensor): depths of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        torch.Tensor: the 3d bounding box tensor :math:`(B, 8, 3)`.

    Examples:
        >>> x_start = torch.tensor([0, 3])
        >>> y_start = torch.tensor([1, 4])
        >>> z_start = torch.tensor([2, 5])
        >>> width = torch.tensor([10, 40])
        >>> height = torch.tensor([20, 50])
        >>> depth = torch.tensor([30, 60])
        >>> bbox_generator3d(x_start, y_start, z_start, width, height, depth)
        tensor([[[ 0,  1,  2],
                 [10,  1,  2],
                 [10, 21,  2],
                 [ 0, 21,  2],
                 [ 0,  1, 32],
                 [10,  1, 32],
                 [10, 21, 32],
                 [ 0, 21, 32]],
        <BLANKLINE>
                [[ 3,  4,  5],
                 [43,  4,  5],
                 [43, 54,  5],
                 [ 3, 54,  5],
                 [ 3,  4, 65],
                 [43,  4, 65],
                 [43, 54, 65],
                 [ 3, 54, 65]]])
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.bbox_generator3d` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.bbox_generator3d instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _bbox_generator3d(x_start, y_start, z_start, width, height, depth)
