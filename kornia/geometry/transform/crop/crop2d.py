from typing import Tuple, Union, Optional

import torch

from kornia.geometry.transform.imgwarp import (
    warp_perspective, get_perspective_transform, warp_affine
)

__all__ = [
    "crop_and_resize",
    "crop_by_boxes",
    "crop_by_transform_mat",
    "center_crop",
    "bbox_to_mask",
    "infer_box_shape",
    "validate_bboxes",
    "bbox_generator",
]


def crop_and_resize(tensor: torch.Tensor, boxes: torch.Tensor, size: Tuple[int, int],
                    mode: str = 'bilinear', padding_mode: str = 'zeros',
                    align_corners: Optional[bool] = None) -> torch.Tensor:
    r"""Extract crops from 2D images (4D tensor) and resize given a bounding box.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (B, C, H, W).
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
            The coordinates would compose a rectangle with a shape of (N1, N2).
        size (Tuple[int, int]): a tuple with the height and width that will be
            used to resize the extracted patches.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool, optional): mode for grid_generation. Default: None.

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
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    if not isinstance(boxes, torch.Tensor):
        raise TypeError("Input boxes type is not a torch.Tensor. Got {}"
                        .format(type(boxes)))

    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # unpack input data
    dst_h, dst_w = size

    # [x, y] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: torch.Tensor = boxes.to(tensor)

    # [x, y] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]], device=tensor.device, dtype=tensor.dtype).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes(tensor, points_src, points_dst, mode, padding_mode, align_corners)


def center_crop(tensor: torch.Tensor, size: Tuple[int, int],
                mode: str = 'bilinear', padding_mode: str = 'zeros',
                align_corners: Optional[bool] = None) -> torch.Tensor:
    r"""Crop the 2D images (4D tensor) from the center.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (B, C, H, W).
        size (Tuple[int, int]): a tuple with the expected height and width
          of the output patch.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool, optional): mode for grid_generation. Default: None.

    Returns:
        torch.Tensor: the output tensor with patches.

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
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))

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
    points_src: torch.Tensor = torch.tensor([[
        [start_x, start_y],
        [end_x, start_y],
        [end_x, end_y],
        [start_x, end_y],
    ]], device=tensor.device, dtype=tensor.dtype)

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]], device=tensor.device, dtype=tensor.dtype).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes(tensor, points_src, points_dst,
                         mode, padding_mode, align_corners)


def crop_by_boxes(tensor: torch.Tensor, src_box: torch.Tensor, dst_box: torch.Tensor,
                  mode: str = 'bilinear', padding_mode: str = 'zeros',
                  align_corners: Optional[bool] = None) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) given two bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width and height.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (B, C, H, W).
        src_box (torch.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        dst_box (torch.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool, optional): mode for grid_generation. Default: None.

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
    validate_bboxes(src_box)
    validate_bboxes(dst_box)

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: torch.Tensor = get_perspective_transform(
        src_box.to(tensor), dst_box.to(tensor))

    bbox: Tuple[torch.Tensor, torch.Tensor] = infer_box_shape(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all(), (
        f"Cropping height, width and depth must be exact same in a batch. "
        f"Got height {bbox[0]} and width {bbox[1]}.")

    h_out: int = int(bbox[0][0].item())
    w_out: int = int(bbox[1][0].item())

    return crop_by_transform_mat(
        tensor, dst_trans_src, (h_out, w_out),
        mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def crop_by_transform_mat(
    tensor: torch.Tensor, transform: torch.Tensor, out_size: Tuple[int, int],
    mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: Optional[bool] = None
) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) given a perspective transformation matrix.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (B, C, H, W).
        transform (torch.Tensor): a perspective transformation matrix with shape (B, 3, 3).
        out_size (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool, optional): mode for grid_generation. Default: None.

    Returns:
        torch.Tensor: the output tensor with patches.
    """
    # simulate broadcasting
    dst_trans_src = torch.as_tensor(
        transform.expand(tensor.shape[0], -1, -1), device=tensor.device, dtype=tensor.dtype)

    patches: torch.Tensor = warp_affine(
        tensor, dst_trans_src[:, :2, :], out_size,
        mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    return patches


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
    validate_bboxes(boxes)
    width: torch.Tensor = (boxes[:, 1, 0] - boxes[:, 0, 0] + 1)
    height: torch.Tensor = (boxes[:, 2, 1] - boxes[:, 0, 1] + 1)
    return (height, width)


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
    if not torch.allclose((boxes[:, 1, 0] - boxes[:, 0, 0] + 1),
                          (boxes[:, 2, 0] - boxes[:, 3, 0] + 1)):
        raise ValueError("Boxes must have be rectangular, while get widths %s and %s" %
                         (str(boxes[:, 1, 0] - boxes[:, 0, 0] + 1),
                          str(boxes[:, 2, 0] - boxes[:, 3, 0] + 1)))

    if not torch.allclose((boxes[:, 2, 1] - boxes[:, 0, 1] + 1),
                          (boxes[:, 3, 1] - boxes[:, 1, 1] + 1)):
        raise ValueError("Boxes must have be rectangular, while get heights %s and %s" %
                         (str(boxes[:, 2, 1] - boxes[:, 0, 1] + 1),
                          str(boxes[:, 3, 1] - boxes[:, 1, 1] + 1)))

    return True


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
    validate_bboxes(boxes)
    # zero padding the surroudings
    mask = torch.zeros((len(boxes), height + 2, width + 2))
    # push all points one pixel off
    # in order to zero-out the fully filled rows or columns
    boxes += 1

    mask_out = []
    # TODO: Looking for a vectorized way
    for m, box in zip(mask, boxes):
        m = m.index_fill(1, torch.arange(box[0, 0].item(), box[1, 0].item() + 1, dtype=torch.long), torch.tensor(1))
        m = m.index_fill(0, torch.arange(box[1, 1].item(), box[2, 1].item() + 1, dtype=torch.long), torch.tensor(1))
        m = m.unsqueeze(dim=0)
        m_out = (m == 1).all(dim=1) * (m == 1).all(dim=2).T
        m_out = m_out[1:-1, 1:-1]
        mask_out.append(m_out)

    return torch.stack(mask_out, dim=0).float()


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
    assert x_start.shape == y_start.shape and x_start.dim() in [0, 1], \
        f"`x_start` and `y_start` must be a scalar or (B,). Got {x_start}, {y_start}."
    assert width.shape == height.shape and width.dim() in [0, 1], \
        f"`width` and `height` must be a scalar or (B,). Got {width}, {height}."
    assert x_start.dtype == y_start.dtype == width.dtype == height.dtype, (
        "All tensors must be in the same dtype. Got "
        f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `width`({width.dtype}), `height`({height.dtype})."
    )
    assert x_start.device == y_start.device == width.device == height.device, (
        "All tensors must be in the same device. Got "
        f"`x_start`({x_start.device}), `y_start`({x_start.device}), `width`({width.device}), `height`({height.device})."
    )

    bbox = torch.tensor([[
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]], device=x_start.device, dtype=x_start.dtype).repeat(1 if x_start.dim() == 0 else len(x_start), 1, 1)

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, 1, 0] += width - 1
    bbox[:, 2, 0] += width - 1
    bbox[:, 2, 1] += height - 1
    bbox[:, 3, 1] += height - 1

    return bbox
