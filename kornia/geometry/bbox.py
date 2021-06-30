from typing import Tuple

import torch

import kornia


@torch.jit.ignore
def validate_bbox(boxes: torch.Tensor) -> bool:
    """Validate if a 2D bounding box usable or not. This function checks if the boxes are rectangular or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right, bottom-right,
            bottom-left. The coordinates must be in the x, y order.
    """
    assert len(boxes.shape) == 3 and boxes.shape[1:] == torch.Size(
        [4, 2]
    ), f"Box shape must be (B, 4, 2). Got {boxes.shape}."

    if not torch.allclose((boxes[:, 1, 0] - boxes[:, 0, 0] + 1), (boxes[:, 2, 0] - boxes[:, 3, 0] + 1)):
        raise ValueError(
            "Boxes must have be rectangular, while get widths %s and %s"
            % (str(boxes[:, 1, 0] - boxes[:, 0, 0] + 1), str(boxes[:, 2, 0] - boxes[:, 3, 0] + 1))
        )

    if not torch.allclose((boxes[:, 2, 1] - boxes[:, 0, 1] + 1), (boxes[:, 3, 1] - boxes[:, 1, 1] + 1)):
        raise ValueError(
            "Boxes must have be rectangular, while get heights %s and %s"
            % (str(boxes[:, 2, 1] - boxes[:, 0, 1] + 1), str(boxes[:, 3, 1] - boxes[:, 1, 1] + 1))
        )

    return True


@torch.jit.ignore
def validate_bbox3d(boxes: torch.Tensor) -> bool:
    """Validate if a 3D bounding box usable or not. This function checks if the boxes are cube or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx8x3, where each box is defined in the following ``clockwise`` order: front-top-left, front-top-right,
            front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right, back-bottom-left.
            The coordinates must be in the x, y, z order.
    """
    assert len(boxes.shape) == 3 and boxes.shape[1:] == torch.Size(
        [8, 3]
    ), f"Box shape must be (B, 8, 3). Got {boxes.shape}."

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = left - right + 1
    assert torch.allclose(
        widths.permute(1, 0), widths[:, 0]
    ), f"Boxes must have be cube, while get different widths {widths}."

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = bot - upper + 1
    assert torch.allclose(
        heights.permute(1, 0), heights[:, 0]
    ), f"Boxes must have be cube, while get different heights {heights}."

    depths = boxes[:, 4:, 2] - boxes[:, :4, 2] + 1
    assert torch.allclose(
        depths.permute(1, 0), depths[:, 0]
    ), f"Boxes must have be cube, while get different depths {depths}."

    return True


def infer_bbox_shape(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right, bottom-right,
            bottom-left. The coordinates must be in the x, y order.

    Returns:
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
        >>> infer_bbox_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))
    """
    validate_bbox(boxes)
    width: torch.Tensor = boxes[:, 1, 0] - boxes[:, 0, 0] + 1
    height: torch.Tensor = boxes[:, 2, 1] - boxes[:, 0, 1] + 1
    return height, width


def infer_bbox_shape3d(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Auto-infer the output sizes for the given 3D bounding boxes.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx8x3, where each box is defined in the following ``clockwise`` order: front-top-left, front-top-right,
            front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right, back-bottom-left.
            The coordinates must be in the x, y, z order.

    Returns:
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
        >>> infer_bbox_shape3d(boxes)
        (tensor([31, 61]), tensor([21, 51]), tensor([11, 41]))
    """
    validate_bbox3d(boxes)

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = (left - right + 1)[:, 0]

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = (bot - upper + 1)[:, 0]

    depths = (boxes[:, 4:, 2] - boxes[:, :4, 2] + 1)[:, 0]
    return depths, heights, widths


def bbox_to_mask(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert 2D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right, bottom-right
            and bottom-left. The coordinates must be in the x, y order.
        width: width of the masked image.
        height: height of the masked image.

    Returns:
        the output mask tensor.

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
    validate_bbox(boxes)
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


def bbox_to_mask3d(boxes: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """Convert 3D bounding boxes to masks. Covered area is 1. and the remaining is 0.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx8x3, where each box is defined in the following ``clockwise`` order: front-top-left, front-top-right,
            front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right, back-bottom-left.
            The coordinates must be in the x, y, z order.
        size: depth, height and width of the masked image.

    Returns:
        the output mask tensor.

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
    validate_bbox3d(boxes)
    mask = torch.zeros((len(boxes), *size))

    mask_out = []
    # TODO: Looking for a vectorized way
    for m, box in zip(mask, boxes):
        m = m.index_fill(
            0,
            torch.arange(box[0, 2].item(), box[4, 2].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.index_fill(
            1,
            torch.arange(box[1, 1].item(), box[2, 1].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.index_fill(
            2,
            torch.arange(box[0, 0].item(), box[1, 0].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.unsqueeze(dim=0)
        m_out = torch.ones_like(m)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=3, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=3, keepdim=True)
        mask_out.append(m_out)

    return torch.stack(mask_out, dim=0).float()


def bbox_generator(
    x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
    """Generate 2D bounding boxes according to the provided start coords, width and height.

    Args:
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        width: widths of the masked image. Shape must be a scalar tensor or :math:`(B,)`.
        height: heights of the masked image. Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        the bounding box tensor.

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
    assert x_start.shape == y_start.shape and x_start.dim() in [
        0,
        1,
    ], f"`x_start` and `y_start` must be a scalar or (B,). Got {x_start}, {y_start}."
    assert width.shape == height.shape and width.dim() in [
        0,
        1,
    ], f"`width` and `height` must be a scalar or (B,). Got {width}, {height}."
    assert x_start.dtype == y_start.dtype == width.dtype == height.dtype, (
        "All tensors must be in the same dtype. Got "
        f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `width`({width.dtype}), `height`({height.dtype})."
    )
    assert x_start.device == y_start.device == width.device == height.device, (
        "All tensors must be in the same device. Got "
        f"`x_start`({x_start.device}), `y_start`({x_start.device}), `width`({width.device}), `height`({height.device})."
    )

    bbox = torch.tensor([[[0, 0], [0, 0], [0, 0], [0, 0]]], device=x_start.device, dtype=x_start.dtype).repeat(
        1 if x_start.dim() == 0 else len(x_start), 1, 1
    )

    bbox[:, :, 0] += x_start.view(-1, 1)
    bbox[:, :, 1] += y_start.view(-1, 1)
    bbox[:, 1, 0] += width - 1
    bbox[:, 2, 0] += width - 1
    bbox[:, 2, 1] += height - 1
    bbox[:, 3, 1] += height - 1

    return bbox


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
        x_start: a tensor containing the x coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        y_start: a tensor containing the y coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        z_start: a tensor containing the z coordinates of the bounding boxes to be extracted. Shape must be a scalar
            tensor or :math:`(B,)`.
        width: widths of the masked image. Shape must be a scalar tensor or :math:`(B,)`.
        height: heights of the masked image. Shape must be a scalar tensor or :math:`(B,)`.
        depth: depths of the masked image. Shape must be a scalar tensor or :math:`(B,)`.

    Returns:
        the 3d bounding box tensor :math:`(B, 8, 3)`.

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
    assert x_start.shape == y_start.shape == z_start.shape and x_start.dim() in [
        0,
        1,
    ], f"`x_start`, `y_start` and `z_start` must be a scalar or (B,). Got {x_start}, {y_start}, {z_start}."
    assert width.shape == height.shape == depth.shape and width.dim() in [
        0,
        1,
    ], f"`width`, `height` and `depth` must be a scalar or (B,). Got {width}, {height}, {depth}."
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
    bbox = torch.tensor(
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], device=x_start.device, dtype=x_start.dtype
    ).repeat(len(x_start), 1, 1)

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


def transform_bbox(trans_mat: torch.Tensor, boxes: torch.Tensor, mode: str = "xyxy") -> torch.Tensor:
    r"""Function that applies a transformation matrix to a box or batch of boxes. Boxes must
    be a tensor of the shape (N, 4) or a batch of boxes (B, N, 4) and trans_mat must be a (3, 3)
    transformation matrix or a batch of transformation matrices (B, 3, 3)

    Args:
        trans_mat: The transformation matrix to be applied
        boxes: The boxes to be transformed
        mode: The format in which the boxes are provided. If set to 'xyxy' the boxes are assumed to be in the format
            ``xmin, ymin, xmax, ymax``. If set to 'xywh' the boxes are assumed to be in the format
            ``xmin, ymin, width, height``
    Returns:
        The set of transformed points in the specified mode


    """
    if not isinstance(mode, str):
        raise TypeError(f"Mode must be a string. Got {type(mode)}")

    if mode not in ("xyxy", "xywh"):
        raise ValueError(f"Mode must be one of 'xyxy', 'xywh'. Got {mode}")

    # convert boxes to format xyxy
    if mode == "xywh":
        boxes[..., -2] = boxes[..., 0] + boxes[..., -2]  # x + w
        boxes[..., -1] = boxes[..., 1] + boxes[..., -1]  # y + h

    transformed_boxes: torch.Tensor = kornia.transform_points(trans_mat, boxes.view(boxes.shape[0], -1, 2))
    transformed_boxes = transformed_boxes.view_as(boxes)

    if mode == 'xywh':
        transformed_boxes[..., 2] = transformed_boxes[..., 2] - transformed_boxes[..., 0]
        transformed_boxes[..., 3] = transformed_boxes[..., 3] - transformed_boxes[..., 1]

    return transformed_boxes
