from typing import Tuple

import torch

from kornia.geometry.transform.projwarp import (
    get_perspective_transform3d, warp_affine3d
)

__all__ = [
    "crop_and_resize3d",
    "crop_by_boxes3d",
    "center_crop3d",
    "bbox_to_mask3d",
    "infer_box_shape3d",
    "validate_bboxes3d",
    "bbox_generator3d"
]


def crop_and_resize3d(tensor: torch.Tensor, boxes: torch.Tensor, size: Tuple[int, int, int],
                      interpolation: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    r"""Extract crops from 3D volumes (5D tensor) and resize them.

    Args:
        tensor (torch.Tensor): the 3D volume tensor with shape (B, C, D, H, W).
        boxes (torch.Tensor): a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        size (Tuple[int, int, int]): a tuple with the height and width that will be
            used to resize the extracted patches.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
            https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details.

    Returns:
        torch.Tensor: tensor containing the patches with shape (Bx)CxN1xN2xN3.

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
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not isinstance(boxes, (torch.Tensor)):
        raise TypeError("Input boxes type is not a torch.Tensor. Got {}"
                        .format(type(boxes)))
    if not isinstance(size, (tuple, list,)) and len(size) != 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}"
                         .format(size))
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
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0, 0],
        [dst_w - 1, 0, 0],
        [dst_w - 1, dst_h - 1, 0],
        [0, dst_h - 1, 0],
        [0, 0, dst_d - 1],
        [dst_w - 1, 0, dst_d - 1],
        [dst_w - 1, dst_h - 1, dst_d - 1],
        [0, dst_h - 1, dst_d - 1],
    ]], dtype=tensor.dtype, device=tensor.device).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes3d(tensor, points_src, points_dst, interpolation, align_corners)


def center_crop3d(tensor: torch.Tensor, size: Tuple[int, int, int],
                  interpolation: str = 'bilinear',
                  align_corners: bool = True) -> torch.Tensor:
    r"""Crop the 3D volumes (5D tensor) at the center.

    Args:
        tensor (torch.Tensor): the 3D volume tensor with shape (B, C, D, H, W).
        size (Tuple[int, int, int]): a tuple with the expected depth, height and width
            of the output patch.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
            https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details.

    Returns:
        torch.Tensor: the output tensor with patches.

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
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))

    assert len(tensor.shape) == 5, f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}."

    if not isinstance(size, (tuple, list,)) and len(size) == 3:
        raise ValueError("Input size must be a tuple/list of length 3. Got {}"
                         .format(size))

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
    points_src: torch.Tensor = torch.tensor([[
        [start_x, start_y, start_z],
        [end_x, start_y, start_z],
        [end_x, end_y, start_z],
        [start_x, end_y, start_z],
        [start_x, start_y, end_z],
        [end_x, start_y, end_z],
        [end_x, end_y, end_z],
        [start_x, end_y, end_z],
    ]], device=tensor.device)

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: torch.Tensor = torch.tensor([[
        [0, 0, 0],
        [dst_w - 1, 0, 0],
        [dst_w - 1, dst_h - 1, 0],
        [0, dst_h - 1, 0],
        [0, 0, dst_d - 1],
        [dst_w - 1, 0, dst_d - 1],
        [dst_w - 1, dst_h - 1, dst_d - 1],
        [0, dst_h - 1, dst_d - 1],
    ]], device=tensor.device).expand(points_src.shape[0], -1, -1)

    return crop_by_boxes3d(tensor,
                           points_src.to(tensor.dtype),
                           points_dst.to(tensor.dtype),
                           interpolation,
                           align_corners)


def crop_by_boxes3d(tensor: torch.Tensor, src_box: torch.Tensor, dst_box: torch.Tensor,
                    interpolation: str = 'bilinear', align_corners: bool = False) -> torch.Tensor:
    """Perform crop transform on 3D volumes (5D tensor) by bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width, height and depth.

    Args:
        tensor (torch.Tensor): the 3D volume tensor with shape (B, C, D, H, W).
        src_box (torch.Tensor): a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        dst_box (torch.Tensor): a tensor with shape (B, 8, 3) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx8x3, where each box is defined in the clockwise
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in x, y, z order.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
            https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for details.

    Returns:
        torch.Tensor: the output tensor with patches.

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
    validate_bboxes3d(src_box)
    validate_bboxes3d(dst_box)

    assert len(tensor.shape) == 5, f"Only tensor with shape (B, C, D, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: torch.Tensor = get_perspective_transform3d(src_box.to(tensor.dtype), dst_box.to(tensor.dtype))
    # simulate broadcasting
    dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1).type_as(tensor)

    bbox = infer_box_shape3d(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all() and (bbox[2] == bbox[2][0]).all(), (
        "Cropping height, width and depth must be exact same in a batch."
        f"Got height {bbox[0]}, width {bbox[1]} and depth {bbox[2]}.")
    patches: torch.Tensor = warp_affine3d(
        tensor, dst_trans_src[:, :3, :],
        # TODO: It will break the grads
        (int(bbox[0][0].item()), int(bbox[1][0].item()), int(bbox[2][0].item())),
        flags=interpolation, align_corners=align_corners)

    return patches


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
    validate_bboxes3d(boxes)

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = (left - right + 1)[:, 0]

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = (bot - upper + 1)[:, 0]

    depths = (boxes[:, 4:, 2] - boxes[:, :4, 2] + 1)[:, 0]
    return (depths, heights, widths)


def validate_bboxes3d(boxes: torch.Tensor) -> None:
    """Validate if a 3D bounding box usable or not.

    This function checks if the boxes are cube or not.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx8x3, where each box is defined in the following (clockwise)
            order: front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left,
            back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in the x, y, z order.
    """
    assert len(boxes.shape) == 3 and boxes.shape[1:] == torch.Size([8, 3]), \
        f"Box shape must be (B, 8, 3). Got {boxes.shape}."

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = (left - right + 1)
    assert torch.allclose(widths.permute(1, 0), widths[:, 0]), \
        f"Boxes must have be cube, while get different widths {widths}."

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = (bot - upper + 1)
    assert torch.allclose(heights.permute(1, 0), heights[:, 0]), \
        f"Boxes must have be cube, while get different heights {heights}."

    depths = (boxes[:, 4:, 2] - boxes[:, :4, 2] + 1)
    assert torch.allclose(depths.permute(1, 0), depths[:, 0]), \
        f"Boxes must have be cube, while get different depths {depths}."


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
    validate_bboxes3d(boxes)
    mask = torch.zeros((len(boxes), *size))

    mask_out = []
    # TODO: Looking for a vectorized way
    for m, box in zip(mask, boxes):
        m = m.index_fill(0, torch.arange(
            box[0, 2].item(), box[4, 2].item() + 1, device=box.device, dtype=torch.long
        ), torch.tensor(1, device=box.device, dtype=box.dtype))
        m = m.index_fill(1, torch.arange(
            box[1, 1].item(), box[2, 1].item() + 1, device=box.device, dtype=torch.long
        ), torch.tensor(1, device=box.device, dtype=box.dtype))
        m = m.index_fill(2, torch.arange(
            box[0, 0].item(), box[1, 0].item() + 1, device=box.device, dtype=torch.long
        ), torch.tensor(1, device=box.device, dtype=box.dtype))
        m = m.unsqueeze(dim=0)
        m_out = torch.ones_like(m)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=3, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=3, keepdim=True)
        mask_out.append(m_out)

    return torch.stack(mask_out, dim=0).float()


def bbox_generator3d(
    x_start: torch.Tensor, y_start: torch.Tensor, z_start: torch.Tensor,
    width: torch.Tensor, height: torch.Tensor, depth: torch.Tensor
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
    assert x_start.shape == y_start.shape == z_start.shape and x_start.dim() in [0, 1], \
        f"`x_start`, `y_start` and `z_start` must be a scalar or (B,). Got {x_start}, {y_start}, {z_start}."
    assert width.shape == height.shape == depth.shape and width.dim() in [0, 1], \
        f"`width`, `height` and `depth` must be a scalar or (B,). Got {width}, {height}, {depth}."
    assert x_start.dtype == y_start.dtype == z_start.dtype == width.dtype == height.dtype == depth.dtype, (
        "All tensors must be in the same dtype. "
        f"Got `x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `z_start`({x_start.dtype}), "
        f"`width`({width.dtype}), `height`({height.dtype}) and `depth`({depth.dtype})."
    )
    assert x_start.device == y_start.device == z_start.device == width.device == height.device == depth.device, (
        "All tensors must be in the same device. "
        f"Got `x_start`({x_start.device}), `y_start`({x_start.device}), `z_start`({x_start.device}), "
        f"`width`({width.device}), `height`({height.device}) and `depth`({depth.device})."
    )

    # front
    bbox = torch.tensor([[
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]], device=x_start.device, dtype=x_start.dtype).repeat(len(x_start), 1, 1)

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, :, 2] += z_start.view(-1, 1)
    bbox[:, 1, 0] += width
    bbox[:, 2, 0] += width
    bbox[:, 2, 1] += height
    bbox[:, 3, 1] += height

    # back
    bbox_back = bbox.clone()
    bbox_back[:, :, -1] += depth.unsqueeze(dim=1).repeat(1, 4)
    bbox = torch.cat([bbox, bbox_back], dim=1)

    return bbox
