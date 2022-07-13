from typing import Optional, Tuple

import torch

from kornia.geometry.bbox import infer_bbox_shape, validate_bbox

from .affwarp import resize
from .imgwarp import get_perspective_transform, warp_affine

__all__ = ["crop_and_resize", "crop_by_boxes", "crop_by_transform_mat", "crop_by_indices", "center_crop"]


def crop_and_resize(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
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
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"Input boxes type is not a torch.Tensor. Got {type(boxes)}")

    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")

    if len(tensor.shape) != 4:
        raise AssertionError(f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}.")

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
    align_corners: bool = True,
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
        raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

    if not isinstance(size, (tuple, list)) and len(size) == 2:
        raise ValueError(f"Input size must be a tuple/list of length 2. Got {size}")

    if len(tensor.shape) != 4:
        raise AssertionError(f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}.")

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
    align_corners: bool = True,
    validate_boxes: bool = True,
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
        validate_boxes: flag to perform validation on boxes.

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
    if validate_boxes:
        validate_bbox(src_box)
        validate_bbox(dst_box)

    if len(tensor.shape) != 4:
        raise AssertionError(f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}.")

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: torch.Tensor = get_perspective_transform(src_box.to(tensor), dst_box.to(tensor))

    bbox: Tuple[torch.Tensor, torch.Tensor] = infer_bbox_shape(dst_box)
    if not ((bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all()):
        raise AssertionError(
            f"Cropping height, width and depth must be exact same in a batch. "
            f"Got height {bbox[0]} and width {bbox[1]}."
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
    align_corners: bool = True,
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


def crop_by_indices(
    input: torch.Tensor,
    src_box: torch.Tensor,
    size: Optional[Tuple] = None,
    interpolation: str = 'bilinear',
    align_corners: Optional[bool] = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Crop tensors with naive indices.

    Args:
        input: the 2D image tensor with shape (B, C, H, W).
        src_box: a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        size: output size. An auto resize will be performed if the cropped slice sizes are not exactly align `size`.
            If None, will auto-infer from src_box.
        interpolation:  algorithm used for upsampling: ``'nearest'`` | ``'linear'`` | ``'bilinear'`` |
            'bicubic' | 'trilinear' | 'area'.
        align_corners: interpolation flag.
        antialias: if True, then image will be filtered with Gaussian before downscaling.
            No effect for upscaling.
    """
    B, C, _, _ = input.shape
    src = torch.as_tensor(src_box, device=input.device, dtype=torch.long)
    x1 = src[:, 0, 0]
    x2 = src[:, 1, 0] + 1
    y1 = src[:, 0, 1]
    y2 = src[:, 3, 1] + 1

    if (
        len(x1.unique(sorted=False))
        == len(x2.unique(sorted=False))
        == len(y1.unique(sorted=False))
        == len(y2.unique(sorted=False))
        == 1
    ):
        out = input[..., y1[0] : y2[0], x1[0] : x2[0]]  # type:ignore
        if size is not None and out.shape[-2:] != size:
            return resize(
                out, size, interpolation=interpolation, align_corners=align_corners, side="short", antialias=antialias
            )

    if size is None:
        h, w = infer_bbox_shape(src)
        size = h.unique(sorted=False), w.unique(sorted=False)
    out = torch.empty(B, C, *size, device=input.device, dtype=input.dtype)
    # Find out the cropped shapes that need to be resized.
    shape_list = torch.stack([y2 - y1, x2 - x1], dim=-1)
    _size = torch.as_tensor(size, device=shape_list.device, dtype=shape_list.dtype)
    same_sized = (shape_list == _size).all(-1)
    for i, same in enumerate(same_sized):
        if not same:
            out[i] = resize(
                input[i : i + 1, :, y1[i] : y2[i], x1[i] : x2[i]],  # type:ignore
                size,
                interpolation=interpolation,
                align_corners=align_corners,
                side="short",
                antialias=antialias,
            )
        else:
            out[i] = input[i : i + 1, :, y1[i] : y2[i], x1[i] : x2[i]]  # type:ignore
    return out
