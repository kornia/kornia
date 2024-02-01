from __future__ import annotations

import warnings
from typing import Optional

import torch

from kornia.core import arange, ones_like, stack, where, zeros

from .linalg import transform_points

__all__ = [
    "validate_bbox",
    "validate_bbox3d",
    "infer_bbox_shape",
    "infer_bbox_shape3d",
    "bbox_to_mask",
    "bbox_to_mask3d",
    "bbox_generator",
    "bbox_generator3d",
    "transform_bbox",
    "nms",
]


def validate_bbox(boxes: torch.Tensor) -> bool:
    """Validate if a 2D bounding box usable or not. This function checks if the boxes are rectangular or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx4x2, where each box is defined in the following ``clockwise`` order: top-left, top-right, bottom-right,
            bottom-left. The coordinates must be in the x, y order.
    """
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([4, 2])):
        return False

    if len(boxes.shape) == 4:
        boxes = boxes.view(-1, 4, 2)

    x_tl, y_tl = boxes[..., 0, 0], boxes[..., 0, 1]
    x_tr, y_tr = boxes[..., 1, 0], boxes[..., 1, 1]
    x_br, y_br = boxes[..., 2, 0], boxes[..., 2, 1]
    x_bl, y_bl = boxes[..., 3, 0], boxes[..., 3, 1]

    width_t, width_b = x_tr - x_tl + 1, x_br - x_bl + 1
    height_t, height_b = y_tr - y_tl + 1, y_br - y_bl + 1

    if not torch.allclose(width_t, width_b, atol=1e-4):
        return False

    if not torch.allclose(height_t, height_b, atol=1e-4):
        return False

    return True


def validate_bbox3d(boxes: torch.Tensor) -> bool:
    """Validate if a 3D bounding box usable or not. This function checks if the boxes are cube or not.

    Args:
        boxes: a tensor containing the coordinates of the bounding boxes to be extracted. The tensor must have the shape
            of Bx8x3, where each box is defined in the following ``clockwise`` order: front-top-left, front-top-right,
            front-bottom-right, front-bottom-left, back-top-left, back-top-right, back-bottom-right, back-bottom-left.
            The coordinates must be in the x, y, z order.
    """
    if not (len(boxes.shape) in [3, 4] and boxes.shape[-2:] == torch.Size([8, 3])):
        raise AssertionError(f"Box shape must be (B, 8, 3) or (B, N, 8, 3). Got {boxes.shape}.")

    if len(boxes.shape) == 4:
        boxes = boxes.view(-1, 8, 3)

    left = torch.index_select(boxes, 1, torch.tensor([1, 2, 5, 6], device=boxes.device, dtype=torch.long))[:, :, 0]
    right = torch.index_select(boxes, 1, torch.tensor([0, 3, 4, 7], device=boxes.device, dtype=torch.long))[:, :, 0]
    widths = left - right + 1
    if not torch.allclose(widths.permute(1, 0), widths[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different widths {widths}.")

    bot = torch.index_select(boxes, 1, torch.tensor([2, 3, 6, 7], device=boxes.device, dtype=torch.long))[:, :, 1]
    upper = torch.index_select(boxes, 1, torch.tensor([0, 1, 4, 5], device=boxes.device, dtype=torch.long))[:, :, 1]
    heights = bot - upper + 1
    if not torch.allclose(heights.permute(1, 0), heights[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different heights {heights}.")

    depths = boxes[:, 4:, 2] - boxes[:, :4, 2] + 1
    if not torch.allclose(depths.permute(1, 0), depths[:, 0]):
        raise AssertionError(f"Boxes must have be cube, while get different depths {depths}.")

    return True


def infer_bbox_shape(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def infer_bbox_shape3d(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # zero padding the surroundings
    mask = zeros((len(boxes), height + 2, width + 2), dtype=boxes.dtype, device=boxes.device)
    # push all points one pixel off
    # in order to zero-out the fully filled rows or columns
    box_i = (boxes + 1).long()
    # set all pixels within box to 1
    for msk, bx in zip(mask, box_i):
        msk[bx[0, 1] : bx[2, 1] + 1, bx[0, 0] : bx[1, 0] + 1] = 1.0
    return mask[:, 1:-1, 1:-1]


def bbox_to_mask3d(boxes: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
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
    mask = zeros((len(boxes), *size))

    mask_out = []
    # TODO: Looking for a vectorized way
    for m, box in zip(mask, boxes):
        m = m.index_fill(
            0,
            arange(box[0, 2].item(), box[4, 2].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.index_fill(
            1,
            arange(box[1, 1].item(), box[2, 1].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.index_fill(
            2,
            arange(box[0, 0].item(), box[1, 0].item() + 1, device=box.device, dtype=torch.long),
            torch.tensor(1, device=box.device, dtype=box.dtype),
        )
        m = m.unsqueeze(dim=0)
        m_out = ones_like(m)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=3, keepdim=True).all(dim=1, keepdim=True)
        m_out = m_out * (m == 1).all(dim=2, keepdim=True).all(dim=3, keepdim=True)
        mask_out.append(m_out)

    return stack(mask_out, dim=0).float()


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
    if not (x_start.shape == y_start.shape and x_start.dim() in [0, 1]):
        raise AssertionError(f"`x_start` and `y_start` must be a scalar or (B,). Got {x_start}, {y_start}.")
    if not (width.shape == height.shape and width.dim() in [0, 1]):
        raise AssertionError(f"`width` and `height` must be a scalar or (B,). Got {width}, {height}.")
    if not x_start.dtype == y_start.dtype == width.dtype == height.dtype:
        raise AssertionError(
            "All tensors must be in the same dtype. Got "
            f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `width`({width.dtype}), `height`({height.dtype})."
        )
    if not x_start.device == y_start.device == width.device == height.device:
        raise AssertionError(
            "All tensors must be in the same device. Got "
            f"`x_start`({x_start.device}), `y_start`({x_start.device}), "
            f"`width`({width.device}), `height`({height.device})."
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
    if not (x_start.shape == y_start.shape == z_start.shape and x_start.dim() in [0, 1]):
        raise AssertionError(
            f"`x_start`, `y_start` and `z_start` must be a scalar or (B,). Got {x_start}, {y_start}, {z_start}."
        )
    if not (width.shape == height.shape == depth.shape and width.dim() in [0, 1]):
        raise AssertionError(f"`width`, `height` and `depth` must be a scalar or (B,). Got {width}, {height}, {depth}.")
    if not x_start.dtype == y_start.dtype == z_start.dtype == width.dtype == height.dtype == depth.dtype:
        raise AssertionError(
            "All tensors must be in the same dtype. "
            f"Got `x_start`({x_start.dtype}), `y_start`({x_start.dtype}), `z_start`({x_start.dtype}), "
            f"`width`({width.dtype}), `height`({height.dtype}) and `depth`({depth.dtype})."
        )
    if not x_start.device == y_start.device == z_start.device == width.device == height.device == depth.device:
        raise AssertionError(
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


def transform_bbox(
    trans_mat: torch.Tensor, boxes: torch.Tensor, mode: str = "xyxy", restore_coordinates: Optional[bool] = None
) -> torch.Tensor:
    r"""Apply a transformation matrix to a box or batch of boxes.

    Args:
        trans_mat: The transformation matrix to be applied with a shape of :math:`(3, 3)`
            or batched as :math:`(B, 3, 3)`.
        boxes: The boxes to be transformed with a common shape of :math:`(N, 4)` or batched as :math:`(B, N, 4)`, the
            polygon shape of :math:`(B, N, 4, 2)` is also supported.
        mode: The format in which the boxes are provided. If set to 'xyxy' the boxes are assumed to be in the format
            ``xmin, ymin, xmax, ymax``. If set to 'xywh' the boxes are assumed to be in the format
            ``xmin, ymin, width, height``
        restore_coordinates: In case the boxes are flipped, adding a post processing step to restore the
            coordinates to a valid bounding box.

    Returns:
        The set of transformed points in the specified mode
    """

    if not isinstance(mode, str):
        raise TypeError(f"Mode must be a string. Got {type(mode)}")

    if mode not in ("xyxy", "xywh"):
        raise ValueError(f"Mode must be one of 'xyxy', 'xywh'. Got {mode}")

    # (B, N, 4, 2) shaped polygon boxes do not need to be restored.
    if restore_coordinates is None and not (boxes.shape[-2:] == torch.Size([4, 2])):
        warnings.warn(
            "Previous behaviour produces incorrect box coordinates if a flip transformation performed on boxes."
            "The previous wrong behaviour has been corrected and will be removed in the future versions."
            "If you wish to keep the previous behaviour, please set `restore_coordinates=False`."
            "Otherwise, set `restore_coordinates=True` as an acknowledgement."
        )

    # convert boxes to format xyxy
    if mode == "xywh":
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]  # x + w
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]  # y + h

    transformed_boxes: torch.Tensor = transform_points(trans_mat, boxes.view(boxes.shape[0], -1, 2))
    transformed_boxes = transformed_boxes.view_as(boxes)

    if (restore_coordinates is None or restore_coordinates) and not (boxes.shape[-2:] == torch.Size([4, 2])):
        restored_boxes = transformed_boxes.clone()
        # In case the boxes are flipped, we ensure it is ordered like left-top -> right-bot points
        restored_boxes[..., 0] = torch.min(transformed_boxes[..., [0, 2]], dim=-1)[0]
        restored_boxes[..., 1] = torch.min(transformed_boxes[..., [1, 3]], dim=-1)[0]
        restored_boxes[..., 2] = torch.max(transformed_boxes[..., [0, 2]], dim=-1)[0]
        restored_boxes[..., 3] = torch.max(transformed_boxes[..., [1, 3]], dim=-1)[0]
        transformed_boxes = restored_boxes

    if mode == "xywh":
        transformed_boxes[..., 2] = transformed_boxes[..., 2] - transformed_boxes[..., 0]
        transformed_boxes[..., 3] = transformed_boxes[..., 3] - transformed_boxes[..., 1]

    return transformed_boxes


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Perform non-maxima suppression (NMS) on a given tensor of bounding boxes according to the intersection-over-
    union (IoU).

    Args:
        boxes: tensor containing the encoded bounding boxes with the shape :math:`(N, (x_1, y_1, x_2, y_2))`.
        scores: tensor containing the scores associated to each bounding box with shape :math:`(N,)`.
        iou_threshold: the throshold to discard the overlapping boxes.

    Return:
        A tensor mask with the indices to keep from the input set of boxes and scores.

    Example:
        >>> boxes = torch.tensor([
        ...     [10., 10., 20., 20.],
        ...     [15., 5., 15., 25.],
        ...     [100., 100., 200., 200.],
        ...     [100., 100., 200., 200.]])
        >>> scores = torch.tensor([0.9, 0.8, 0.7, 0.9])
        >>> nms(boxes, scores, iou_threshold=0.8)
        tensor([0, 3, 1])
    """
    if len(boxes.shape) != 2 and boxes.shape[-1] != 4:
        raise ValueError(f"boxes expected as Nx4. Got: {boxes.shape}.")

    if len(scores.shape) != 1:
        raise ValueError(f"scores expected as N. Got: {scores.shape}.")

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores mus have same shape. Got: {boxes.shape, scores.shape}.")

    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    if len(keep) > 0:
        return stack(keep)

    return torch.tensor(keep)
