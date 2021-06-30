import warnings
from typing import Optional, Tuple

import torch

from kornia.geometry.bbox import bbox_generator as _bbox_generator
from kornia.geometry.bbox import bbox_to_mask as _bbox_to_mask
from kornia.geometry.bbox import infer_bbox_shape as _infer_bbox_shape
from kornia.geometry.bbox import validate_bbox as _validate_bbox
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_affine

__all__ = [
    "crop_and_resize",
    "crop_by_boxes",
    "crop_by_transform_mat",
    "center_crop",
    "validate_bboxes",
    "infer_box_shape",
    "bbox_to_mask",
    "bbox_generator",
]


def crop_and_resize(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Extract crops from 2D images (4D tensor) and resize given a bounding box.

    Args:
        tensor: the 2D image tensor with shape (B, C, H, W).
        boxes : a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
            The coordinates would compose a rectangle with a shape of (N1, N2).
        size: a tuple with the height and width that will be
            used to resize the extracted patches.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | 'reflection'.
        align_corners: mode for grid_generation.

    Returns:
        torch.Tensor: tensor containing the patches with shape BxCxN1xN2.

    Example:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ... ]]])
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> crop_and_resize(input, boxes, (2, 2), mode='nearest', align_corners=True)
        tensor([[[[ 6.,  7.],
                  [10., 11.]]]])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes type is not a torch.Tensor. Got {}".format(type(boxes)))

    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}".format(size))

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # unpack input data
    dst_h, dst_w = size

    # [x, y] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = boxes.to(tensor)

    # [x, y] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], device=tensor.device, dtype=tensor.dtype
    ).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes(tensor, points_src, points_dst, mode, padding_mode, align_corners)


def center_crop(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    r"""Crop the 2D images (4D tensor) from the center.

    Args:
        tensor: the 2D image tensor with shape (B, C, H, W).
        size: a tuple with the expected height and width
          of the output patch.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the output tensor with patches.

    Examples:
        >>> input = torch.tensor([[[
        ...     [1., 2., 3., 4.],
        ...     [5., 6., 7., 8.],
        ...     [9., 10., 11., 12.],
        ...     [13., 14., 15., 16.],
        ...  ]]])
        >>> center_crop(input, (2, 4), mode='nearest', align_corners=True)
        tensor([[[[ 5.,  6.,  7.,  8.],
                  [ 9., 10., 11., 12.]]]])
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))

    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}".format(size))

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = tensor.shape[-2:]

    # compute start/end offsets
    dst_h_half: float = dst_h / 2
    dst_w_half: float = dst_w / 2
    src_h_half: float = src_h / 2
    src_w_half: float = src_w / 2

    start_x: float = src_w_half - dst_w_half
    start_y: float = src_h_half - dst_h_half

    end_x: float = start_x + dst_w - 1
    end_y: float = start_y + dst_h - 1

    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = torch.tensor(
        [[[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]],
        device=tensor.device,
        dtype=tensor.dtype,
    )

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor(
        [[[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]]], device=tensor.device, dtype=tensor.dtype
    ).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes(tensor, points_src, points_dst, mode, padding_mode, align_corners)


def crop_by_boxes(
    tensor: torch.Tensor,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) given two bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width and height.

    Args:
        tensor: the 2D image tensor with shape (B, C, H, W).
        src_box: a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        dst_box: a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        torch.Tensor: the output tensor with patches.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
        >>> src_box = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> dst_box = torch.tensor([[
        ...     [0., 0.],
        ...     [1., 0.],
        ...     [1., 1.],
        ...     [0., 1.],
        ... ]])  # 1x4x2
        >>> crop_by_boxes(input, src_box, dst_box, align_corners=True)
        tensor([[[[ 5.0000,  6.0000],
                  [ 9.0000, 10.0000]]]])

    Note:
        If the src_box is smaller than dst_box, the following error will be thrown.
        RuntimeError: solve_cpu: For batch 0: U(2,2) is zero, singular U.
    """
    # TODO: improve this since might slow down the function
    _validate_bbox(src_box)
    _validate_bbox(dst_box)

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: torch.Tensor = get_perspective_transform(src_box.to(tensor), dst_box.to(tensor))

    bbox: Tuple[torch.Tensor, torch.Tensor] = _infer_bbox_shape(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all(), (
        f"Cropping height, width and depth must be exact same in a batch. " f"Got height {bbox[0]} and width {bbox[1]}."
    )

    h_out: int = int(bbox[0][0].item())
    w_out: int = int(bbox[1][0].item())

    return crop_by_transform_mat(
        tensor, dst_trans_src, (h_out, w_out), mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def crop_by_transform_mat(
    tensor: torch.Tensor,
    transform: torch.Tensor,
    out_size: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) given a perspective transformation matrix.

    Args:
        tensor: the 2D image tensor with shape (B, C, H, W).
        transform: a perspective transformation matrix with shape (B, 3, 3).
        out_size: size of the output image (height, width).
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode (str): padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: mode for grid_generation.

    Returns:
        the output tensor with patches.
    """
    # simulate broadcasting
    dst_trans_src = torch.as_tensor(transform.expand(tensor.shape[0], -1, -1), device=tensor.device, dtype=tensor.dtype)

    patches: torch.Tensor = warp_affine(
        tensor, dst_trans_src[:, :2, :], out_size, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    return patches


@torch.jit.ignore
def validate_bboxes(boxes: torch.Tensor) -> bool:
    """Validate if a 2D bounding box usable or not.
    This function checks if the boxes are rectangular or not.
    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.

    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.validate_bboxes` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.validate_bbox instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _validate_bbox(boxes)


def infer_box_shape(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - Bounding box heights, shape of :math:`(B,)`.
        - Boundingbox widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]])  # 2x4x2
        >>> infer_box_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.infer_box_shape` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.infer_bbox_shape instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _infer_bbox_shape(boxes)


def bbox_to_mask(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
        width (int): width of the masked image.
        height (int): height of the masked image.

    Returns:
        torch.Tensor: the output mask tensor.

    Note:
        It is currently non-differentiable.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.bbox_to_mask` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.bbox_to_mask instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _bbox_to_mask(boxes, width, height)


def bbox_generator(
    x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    """Generate 2D bounding boxes according to the provided start coords, width and height.

    Args:
        x_start (torch.Tensor): a tensor containing the x coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        y_start (torch.Tensor): a tensor containing the y coordinates of the bounding boxes to be extracted.
            Shape must be a scalar tensor or :math:`(B,)`.
        width (torch.Tensor): widths of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.
        height (torch.Tensor): heights of the masked image.
            Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        torch.Tensor: the bounding box tensor.

    Examples:
        >>> x_start = torch.tensor([0, 1])
        >>> y_start = torch.tensor([1, 0])
        >>> width = torch.tensor([5, 3])
        >>> height = torch.tensor([7, 4])
        >>> bbox_generator(x_start, y_start, width, height)
        tensor([[[0, 1],
                 [4, 1],
                 [4, 7],
                 [0, 7]],
        <BLANKLINE>
                [[1, 0],
                 [3, 0],
                 [3, 3],
                 [1, 3]]])
    """
    warnings.warn(
        "`kornia.geometry.transforms.crop.crop2d.bbox_generator` is deprecated and will be removed > 0.6.0. "
        "Please use `kornia.geometry.bbox.bbox_generator instead.`",
        DeprecationWarning,
        stacklevel=2,
    )
    return _bbox_generator(x_start, y_start, width, height)
