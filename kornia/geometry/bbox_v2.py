from typing import Tuple

import torch

import kornia


def _check_bbox_dimensions(boxes: torch.Tensor):
    if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (4, 2)):
        raise ValueError(f"BBox shape must be (N, 4, 2) or (B, N, 4, 2). Got {boxes.shape}.")


def _check_bbox3d_dimensions(boxes: torch.Tensor):
    if not (3 <= boxes.ndim <= 4 and boxes.shape[-2:] == (8, 3)):
        raise ValueError(f"3D bbox shape must be (N, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")


@torch.jit.ignore
def validate_bbox(boxes: torch.Tensor) -> bool:
    """Validate if a 2D bounding box usable or not. This function checks if the boxes are rectangular or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx4x2 or BxNx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right, ottom-left. The coordinates must be in the x, y order. The height and width of a box with
            corners (x1, y1) and (x2, y2) is computed as ``width = x2 - x1`` and ``height = y2 - y1``.
    """
    _check_bbox_dimensions(boxes)
    if not boxes.is_floating_point():
        raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}")

    if not torch.allclose((boxes[..., 1, 0] - boxes[..., 0, 0]), (boxes[..., 2, 0] - boxes[..., 3, 0])):
        raise ValueError(
            "Boxes must have be rectangular, while get widths %s and %s"
            % (str(boxes[..., 1, 0] - boxes[..., 0, 0]), str(boxes[..., 2, 0] - boxes[..., 3, 0]))
        )

    if not torch.allclose((boxes[..., 2, 1] - boxes[..., 0, 1]), (boxes[..., 3, 1] - boxes[..., 1, 1])):
        raise ValueError(
            "Boxes must have be rectangular, while get heights %s and %s"
            % (str(boxes[..., 2, 1] - boxes[..., 0, 1]), str(boxes[..., 3, 1] - boxes[..., 1, 1]))
        )

    return True


@torch.jit.ignore
def validate_bbox3d(boxes: torch.Tensor) -> bool:
    """Validate if a 3D bounding box usable or not. This function checks if the boxes are cube or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx8x3 or BxNx8x3, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order. The height, width and depth of a 3D box with
            corners (x1, y1, z1) and (x2, y2, z2) is computed as ``width = x2 - x1``, ``height = y2 - y1`` and
            ``depth = z2 - z1``.
    """
    _check_bbox3d_dimensions(boxes)
    if not boxes.is_floating_point():
        raise ValueError(f"Coordinates must be in floating point. Got {boxes.dtype}")

    left = boxes[..., [1, 2, 5, 6], 0]
    right = boxes[..., [0, 3, 4, 7], 0]
    widths = left - right
    if not torch.allclose(widths.permute(1, 0), widths[:, 0]):
        raise ValueError(f"Boxes must have be cube, while get different widths {widths}.")

    bottom = boxes[..., [2, 3, 6, 7], 1]
    upper = boxes[..., [0, 1, 4, 5], 1]
    heights = bottom - upper
    if not torch.allclose(heights.permute(1, 0), heights[:, 0]):
        raise ValueError(f"Boxes must have be cube, while get different heights {heights}.")

    depths = boxes[..., 4:, 2] - boxes[..., :4, 2]
    if not torch.allclose(depths.permute(1, 0), depths[:, 0]):
        ValueError(f"Boxes must have be cube, while get different depths {depths}.")

    return True


def infer_bbox_shape(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx4x2 or BxNx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right, bottom-left. The coordinates must be in the x, y order. The height and width of a box with
            corners (x1, y1) and (x2, y2) is computed as ``width = x2 - x1`` and ``height = y2 - y1``.

    Returns:
        - Bounding box heights, shape of :math:`(N,)` or :math:`(B,N)`.
        - Boundingbox widths, shape of :math:`(N,)` or :math:`(B,N)`.

    Example:
        >>> boxes = torch.tensor([[[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]]])  # 1x2x4x2
        >>> infer_bbox_shape(boxes)
        (tensor([[1., 1.]]), tensor([[1., 2.]]))
    """
    _check_bbox_dimensions(boxes)
    width: torch.Tensor = boxes[..., 1, 0] - boxes[..., 0, 0]
    height: torch.Tensor = boxes[..., 2, 1] - boxes[..., 0, 1]
    return height, width


def infer_bbox3d_shape(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 3D bounding boxes.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx8x3 or BxNx8x3, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order. The height, width and depth of a 3D box with
            corners (x1, y1, z1) and (x2, y2, z2) is computed as ``width = x2 - x1``, ``height = y2 - y1`` and
            ``depth = z2 - z1``.

    Returns:
        - Bounding box depths, shape of :math:`(N,)` or :math:`(B,)`.
        - Bounding box heights, shape of :math:`(N,)` or :math:`(B,)`.
        - Bounding box widths, shape of :math:`(N,)` or :math:`(B,)`.

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
        >>> infer_bbox3d_shape(boxes)
        (tensor([30, 60]), tensor([20, 50]), tensor([10, 40]))
    """
    _check_bbox3d_dimensions(boxes)

    left = boxes[..., [1, 2, 5, 6], 0]
    right = boxes[..., [0, 3, 4, 7], 0]
    widths = left - right

    bottom = boxes[..., [2, 3, 6, 7], 1]
    upper = boxes[..., [0, 1, 4, 5], 1]
    heights = bottom - upper

    depths = boxes[..., 4:, 2] - boxes[..., :4, 2]
    return depths[..., 0], heights[..., 0], widths[..., 0]


def bbox_to_mask(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx4x2 or BxNx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right,
            bottom-right and bottom-left. The coordinates must be in the x, y order. The height and width of a box with
            corners (x1, y1) and (x2, y2) is computed as ``width = x2 - x1`` and ``height = y2 - y1``.
        width: width of the masked image/images.
        height: height of the masked image/images.

    Returns:
        the output mask tensor, shape of :math:`(N, width, height)` or :math:`(B,N, width, height)` and dtype of boxes.

    Note:
        It is currently non-differentiable.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [4., 1.],
        ...        [4., 3.],
        ...        [1., 3.],
        ...   ]])  # 1x4x2
        >>> bbox_to_mask(boxes, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 1., 1., 1., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])
    """
    _check_bbox_dimensions(boxes)
    mask = torch.zeros((*boxes.shape[:-2], height, width), dtype=boxes.dtype, device=boxes.device)

    # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
    clamped_boxes = boxes.detach().clone()
    clamped_boxes[..., 0].clamp_(0, width)
    clamped_boxes[..., 1].clamp_(0, height)

    boxes_xyxy = kornia_bbox_to_bbox(boxes, mode='xyxy')
    # Reshape mask to (BxN, H, W) and boxes to (BxN, 4) to iterate over all of them.
    # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
    for mask_channel, box_xyxy in zip(mask.view(-1, *mask.shape[-2:]), boxes_xyxy.view(-1, 4).round().int()):
        # Mask channel dimensions: (height, width)
        mask_channel[box_xyxy[1] : box_xyxy[3], box_xyxy[0] : box_xyxy[2]] = 1

    return mask


def bbox3d_to_mask3d(boxes: torch.Tensor, depth: int, width: int, height: int) -> torch.Tensor:
    """Convert 3D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Nx8x3 or BxNx8x3, where each box is defined in the following ``clockwise`` order: front-top-left,
            front-top-right, ront-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right,
            back-bottom-left. The coordinates must be in the x, y, z order. The height, width and depth of a 3D box with
            corners (x1, y1, z1) and (x2, y2, z2) is computed as ``width = x2 - x1``, ``height = y2 - y1`` and
            ``depth = z2 - z1``.
        width: width of the masked image/images.
        height: height of the masked image/images.
        depth: depth of the masked image/images.

    Returns:
        the output mask tensor, shape of :math:`(N, depth, width, height)` or :math:`(B,N, depth, width, height)` and
         dtype of boxes.

    Examples:
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
        >>> bbox3d_to_mask3d(boxes, 4, 5, 5)
        tensor([[[0., 0., 0., 0., 0.],
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
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                [[0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]])
    """
    _check_bbox3d_dimensions(boxes)

    mask = torch.zeros((*boxes.shape[:-3], depth, height, width), dtype=boxes.dtype, device=boxes.device)

    # Boxes coordinates can be outside the image size after transforms. Clamp values to the image size
    clamped_boxes = boxes.detach().clone()
    clamped_boxes[..., 0].clamp_(0, width)
    clamped_boxes[..., 1].clamp_(0, height)
    clamped_boxes[..., 2].clamp_(0, depth)

    boxes_xyzxyz = kornia_bbox3d_to_bbox3d(boxes, mode='xyzxyz')
    # Reshape mask to (BxN, D, H, W) and boxes to (BxN, 6) to iterate over all of them.
    # Cast boxes coordinates to be integer to use them as indexes. Use round to handle decimal values.
    for mask_channel, box_xyzxyz in zip(mask.view(-1, *mask.shape[-3:]), boxes_xyzxyz.view(-1, 6).round().int()):
        # Mask channel dimensions: (depth, height, width)
        mask_channel[box_xyzxyz[2] : box_xyzxyz[5], box_xyzxyz[1] : box_xyzxyz[4], box_xyzxyz[0] : box_xyzxyz[3]] = 1

    return mask


def bbox_to_kornia_bbox(boxes: torch.Tensor, mode: str = "xyxy") -> torch.Tensor:
    r"""Convert 2D bounding boxes to kornia format according to the format in which the boxes are provided.

    Kornia bounding boxes format is defined as a a floating data type tensor of shape Nx4x2 or BxNx4x2, where each box
    is defined in the following ``clockwise`` order: top-left, top-right, bottom-right, bottom-left. The coordinates
    must be in the x, y order. The height and width of a box with corners (x1, y1) and (x2, y2) is computed as
    ``width = x2 - x1`` and ``height = y2 - y1``.

    Args:
        boxes: boxes to be transformed, shape of :math:`(N,4)` or :math:`(B,N,4)`.
        mode: The format in which the boxes are provided.
            - 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin`` and
                ``height = ymax - ymin``.
            - 'xyxy_plus_1': like 'xyxy' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
            - 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where ``width = xmax - xmin``
                and ``height = ymax - ymin``.
            - 'xywh_plus_1': like 'xywh' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
    Returns:
        Bounding boxes tensor in kornia format, shape of :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)` and dtype of
            ``boxes`` if it's a floating point data type and ``float`` if not.

    Examples:
        >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
        >>> bbox_to_kornia_bbox(boxes_xyxy, mode='xyxy')
        tensor([[[0., 3.],
                 [1., 3.],
                 [1., 4.],
                 [0., 4.]],
        <BLANKLINE>
                [[5., 1.],
                 [8., 1.],
                 [8., 4.],
                 [5., 4.]]])

    """
    if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 4):
        raise ValueError(f"BBox shape must be (N, 4) or (B, N, 4). Got {boxes.shape}.")

    boxes = boxes if boxes.is_floating_point() else boxes.float()

    xmin, ymin = boxes[..., 0], boxes[..., 1]
    if mode in ("xyxy", "xyxy_plus_1"):
        height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
    elif mode in ("xywh", "xywh_plus_1"):
        height, width = boxes[..., 3], boxes[..., 2]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if "plus_1" in mode:
        height -= 1
        width -= 1

    # Create (B,N,4,2) or (N,4,2) with all points in top left position of the bounding box
    kornia_boxes = torch.zeros((*boxes.shape[:-1], 4, 2), device=boxes.device, dtype=boxes.dtype)
    kornia_boxes[..., 0] = xmin.unsqueeze(-1)
    kornia_boxes[..., 1] = ymin.unsqueeze(-1)
    # Shift top-right, bottom-right, bottom-left points to the right coordinates
    kornia_boxes[..., 1, 0] += width  # Top right
    kornia_boxes[..., 2, 0] += width  # Bottom right
    kornia_boxes[..., 2, 1] += height  # Bottom right
    kornia_boxes[..., 3, 1] += height  # Bottom left

    return kornia_boxes


def bbox3d_to_kornia_bbox3d(boxes: torch.Tensor, mode: str = "xyzxyz") -> torch.Tensor:
    r"""Convert 3D bounding boxes to kornia format according to the format in which the boxes are provided.

    Kornia 3D bounding boxes format is defined as a floating data type tensor of shape Nx8x3 or BxNx8x3, where each 3D
    box is defined in the following ``clockwise`` order: front-top-left, front-top-right, ront-bottom-right,
    front-bottom-left, ack-top-left, back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in
    the x, y, z order. The height, width and depth of a 3D box with corners (x1, y1, z1) and (x2, y2, z2) is computed as
    ``width = x2 - x1``, ``height = y2 - y1`` and ``depth = z2 - z1``.

    Args:
        boxes: 3D boxes to be transformed, shape of :math:`(N,6)` or :math:`(B,N,6)`.
        mode: The format in which the boxes are provided.
            - 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
            - 'xyzxyz_plus_1': like 'xyzxyz' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                ``depth = zmax - zmin + 1``.
            - 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
            - 'xyzwhd_plus_1': like 'xyzwhd' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                ``depth = zmax - zmin + 1``.
    Returns:
        3D bounding boxes tensor in kornia format, shape of :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` and dtype of
            ``boxes`` if it's a floating point data type and ``float`` if not.

    Examples:
        >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
        >>> bbox3d_to_kornia_bbox3d(boxes_xyzxyz, mode='xyzxyz')
        tensor([[[0., 3., 6.],
                 [1., 3., 6.],
                 [1., 4., 6.],
                 [0., 4., 6.],
                 [0., 3., 8.],
                 [1., 3., 8.],
                 [1., 4., 8.],
                 [0., 4., 8.]],
        <BLANKLINE>
                [[5., 1., 3.],
                 [8., 1., 3.],
                 [8., 4., 3.],
                 [5., 4., 3.],
                 [5., 1., 9.],
                 [8., 1., 9.],
                 [8., 4., 9.],
                 [5., 4., 9.]]])
    """
    if not (2 <= boxes.ndim <= 3 and boxes.shape[-1] == 6):
        raise ValueError(f"BBox shape must be (N, 6) or (B, N, 6). Got {boxes.shape}.")

    boxes = boxes if boxes.is_floating_point() else boxes.float()

    xmin, ymin, zmin = boxes[..., 0], boxes[..., 1], boxes[..., 2]
    if mode in ("xyzxyz", "xyzxyz_plus_1"):
        width = boxes[..., 3] - boxes[..., 0]
        height = boxes[..., 4] - boxes[..., 1]
        depth = boxes[..., 5] - boxes[..., 2]
    elif mode in ("xyzwhd", "xyzwhd_plus_1"):
        depth, height, width = boxes[4], boxes[..., 3], boxes[5]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if "plus_1" in mode:
        height -= 1
        width -= 1
        depth -= 1

    # Front
    # Create (B,N,4,3) or (N,4,3) with all points in top left position of the bounding box
    kornia_front_boxes = torch.zeros((*boxes.shape[:-1], 4, 3), device=boxes.device, dtype=boxes.dtype)
    kornia_front_boxes[..., 0] = xmin.unsqueeze(-1)
    kornia_front_boxes[..., 1] = ymin.unsqueeze(-1)
    kornia_front_boxes[..., 2] = zmin.unsqueeze(-1)
    # Shift front-top-right, front-bottom-right, front-bottom-left points to the right coordinates
    kornia_front_boxes[..., 1, 0] += width  # Top right
    kornia_front_boxes[..., 2, 0] += width  # Bottom right
    kornia_front_boxes[..., 2, 1] += height  # Bottom right
    kornia_front_boxes[..., 3, 1] += height  # Bottom left

    # Back
    kornia_back_boxes = kornia_front_boxes.clone()
    kornia_back_boxes[..., 2] += depth.unsqueeze(-1)

    return torch.cat([kornia_front_boxes, kornia_back_boxes], dim=-1)


def kornia_bbox_to_bbox(kornia_boxes: torch.Tensor, mode: str = "xyxy") -> torch.Tensor:
    r"""Convert 2D bounding boxes in kornia format to the format specified by ``mode``.

    Kornia bounding boxes format is defined as a a floating data type tensor of shape Nx4x2 or BxNx4x2, where each box
    is defined in the following ``clockwise`` order: top-left, top-right, bottom-right, bottom-left. The coordinates
    must be in the x, y order. The height and width of a box with corners (x1, y1) and (x2, y2) is computed as
    ``width = x2 - x1`` and ``height = y2 - y1``.

    Args:
        kornia_boxes: boxes to be transformed, shape of :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
        mode: The format in which the boxes are provided.
            - 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin`` and
                ``height = ymax - ymin``.
            - 'xyxy_plus_1': like 'xyxy' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
            - 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where ``width = xmax - xmin``
                and ``height = ymax - ymin``.
            - 'xywh_plus_1': like 'xywh' where ``width = xmax - xmin + 1`` and  ``height = ymax - ymin + 1``.
    Returns:
        Bounding boxes tensor the ``mode`` format, shape of :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.

    Examples:
        >>> boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
        >>> kornia_bbox = bbox_to_kornia_bbox(boxes_xyxy, mode='xyxy')
        >>> assert (kornia_bbox_to_bbox(kornia_bbox, mode='xyxy') == boxes_xyxy).all()
    """
    _check_bbox_dimensions(kornia_boxes)

    boxes = torch.stack([kornia_boxes.min(dim=-2).values, kornia_boxes.max(dim=-2).values], dim=1).view(
        *kornia_boxes.shape[:-2], 4
    )

    if mode in ("xyxy", "xyxy_plus_1"):
        pass
    elif mode in ("xywh", "xywh_plus_1"):
        height, width = boxes[..., 3] - boxes[..., 1], boxes[..., 2] - boxes[..., 0]
        boxes[..., 2] = width
        boxes[..., 3] = height
    else:
        raise ValueError(f"Unknown mode {mode}")

    if "plus_1" in mode:
        offset = torch.as_tensor([0, 0, 1, 1], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes + offset

    return boxes


def kornia_bbox3d_to_bbox3d(kornia_boxes: torch.Tensor, mode: str = "xyzxyz") -> torch.Tensor:
    r"""Convert 3D bounding boxes in kornia format according to the format specified by ``mode``.

    Kornia 3D bounding boxes format is defined as a floating data type tensor of shape Nx8x3 or BxNx8x3, where each 3D
    box is defined in the following ``clockwise`` order: front-top-left, front-top-right, ront-bottom-right,
    front-bottom-left, ack-top-left, back-top-right, back-bottom-right, back-bottom-left. The coordinates must be in
    the x, y, z order. The height, width and depth of a 3D box with corners (x1, y1, z1) and (x2, y2, z2) is computed as
    ``width = x2 - x1``, ``height = y2 - y1`` and ``depth = z2 - z1``.

    Args:
        kornia_boxes: 3D boxes to be transformed, shape of :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
        mode: The format in which the boxes are provided.
            - 'xyzxyz': boxes are assumed to be in the format ``xmin, ymin, zmin, xmax, ymax, zmax`` where
                ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
            - 'xyzxyz_plus_1': like 'xyzxyz' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                ``depth = zmax - zmin + 1``.
            - 'xyzwhd': boxes are assumed to be in the format ``xmin, ymin, zmin, width, height, depth`` where
                ``width = xmax - xmin``, ``height = ymax - ymin`` and ``depth = zmax - zmin``.
            - 'xyzwhd_plus_1': like 'xyzwhd' where ``width = xmax - xmin + 1``, ``height = ymax - ymin + 1`` and
                ``depth = zmax - zmin + 1``.
    Returns:
        3D bounding boxes tensor the ``mode`` format, shape of :math:`(N, 6)` or :math:`(B, N, 6)`.


    Examples:
        >>> boxes_xyzxyz = torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]])
        >>> kornia_bbox = bbox3d_to_kornia_bbox3d(boxes_xyzxyz, mode='xyzxyz')
        >>> assert (kornia_bbox3d_to_bbox3d(kornia_bbox, mode='xyzxyz') == boxes_xyzxyz).all()
    """
    _check_bbox3d_dimensions(kornia_boxes)
    boxes = torch.stack([kornia_boxes.min(dim=-2).values, kornia_boxes.max(dim=-2).values], dim=1).view(
        *kornia_boxes.shape[:-2], 6
    )

    if mode in ("xyzxyz", "xyzxyz_plus_1"):
        pass
    elif mode in ("xyzwhd", "xyzwhd_plus_1"):
        width = boxes[..., 3] - boxes[..., 0]
        height = boxes[..., 4] - boxes[..., 1]
        depth = boxes[..., 5] - boxes[..., 2]
        boxes[..., 3] = width
        boxes[..., 4] = height
        boxes[..., 5] = depth
    else:
        raise ValueError(f"Unknown mode {mode}")

    if "plus_1" in mode:
        offset = torch.as_tensor([0, 0, 0, 1, 1, 1], device=boxes.device, dtype=boxes.dtype)
        boxes = boxes + offset

    return boxes


def _transform_bbox(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Transforms 3D and 2D in kornia format by applying the transformation matrix M. Boxes and the transformation matrix
    could be batched or not.

    Args:
        boxes: 2D or 3D boxes in kornia format.
        M: the transformation matrix of shape :math:`(3, 3)` or :math:`(B, 3, 3)` for 2D and :math:`(4, 4)` or
            :math:`(B, 4, 4)` for 3D boxes.
    """
    boxes = boxes if boxes.is_floating_point() else boxes.float()
    M = M if M.is_floating_point() else M.float()

    # Work with batch as kornia.transform_points only supports a batch of points.
    boxes_per_batch, n_points_per_box, coordinates_dimension = boxes.shape[-3:]
    points = boxes.view(-1, n_points_per_box * boxes_per_batch, coordinates_dimension)
    M = M if M.ndim == 3 else M.unsqueeze(0)

    if points.shape[0] != M.shape[0]:
        raise ValueError(
            f"Batch size mismatch. Got {points.shape[0]} for boxes, {M.shape[0]} for the transformation matrix."
        )

    transformed_boxes: torch.Tensor = kornia.transform_points(M, points)
    transformed_boxes = transformed_boxes.view_as(boxes)
    return transformed_boxes


def transform_bbox(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    r"""Function that applies a transformation matrix to boxes or a batch of boxes in kornia format. That it's, a
    floating data type tensor of shape Nx4x2 or BxNx4x2, where each box is defined in the following ``clockwise`` order:
    top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order. The height and width
    of a box with corners (x1, y1) and (x2, y2) is computed as ``width = x2 - x1`` and ``height = y2 - y1``.

    Args:
        boxes: The boxes to be transformed. It must be a tensor of shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
        M: The transformation matrix to be applied, shape of :math:`(3, 3)` or :math:`(B, 3, 3)`.
    Returns:
        A tensor of shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)` with the transformed boxes in kornia format.
    """
    _check_bbox_dimensions(boxes)
    if not 2 <= M.ndim <= 3 or M.shape[-2:] != (3, 3):
        raise ValueError(f"The transformation matrix shape must be (3, 3) or (B, 3, 3). Got {M.shape}.")

    return _transform_bbox(boxes, M)


def transform_bbox3d(boxes: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    r"""Function that applies a transformation matrix to 3D boxes or a batch of 3D boxes in kornia format. That it's, a
    floating data type tensor of shape Nx8x3 or BxNx8x3, where each box is defined in the following ``clockwise`` order:
    front-top-left, front-top-right, front-bottom-right, front-bottom-left, back-top-left, back-top-right,
    back-bottom-right,back-bottom-left. The coordinates must be in the x, y, z order. The height, width and depth of a
    3D box with corners (x1, y1, z1) and (x2, y2, z2) is computed as ``width = x2 - x1``, ``height = y2 - y1`` and
    ``depth = z2 - z1``.

    Args:
        boxes: The 3D boxes to be transformed. It must be a tensor of shape :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)`.
        M: The transformation matrix to be applied, shape of :math:`(4, 4)` or :math:`(B, 4, 4)`.
    Returns:
        A tensor of shape :math:`(N, 8, 3)` or :math:`(B, N, 8, 3)` with the transformed 3D boxes in kornia format.
    """
    _check_bbox3d_dimensions(boxes)
    if not 2 <= M.ndim <= 3 or M.shape[-2:] != (4, 4):
        raise ValueError(f"The transformation matrix shape must be (4, 4) or (B, 4, 4). Got {M.shape}.")

    return _transform_bbox(boxes, M)
