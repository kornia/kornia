from typing import Optional

import kornia
import torch
from .crop2d import crop_by_boxes as sampling_by_boxes, validate_bboxes


def generate_dst_bbox_from_bbox(boxes: torch.Tensor):
    """Generate 2D destination bounding boxes from bounding boxes. Preserving the exact width and height.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.

    Returns:
        torch.Tensor: the output mask tensor.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [4., 1.],
        ...        [4., 2.],
        ...        [1., 2.],
        ...   ], [
        ...        [1., 3.],
        ...        [8., 3.],
        ...        [8., 7.],
        ...        [1., 7.],
        ...   ]])  # 1x4x2
        >>> generate_dst_bbox_from_bbox(boxes)
        tensor([[[0., 0.],
                 [3., 0.],
                 [3., 1.],
                 [0., 1.]],
        <BLANKLINE>
                [[0., 0.],
                 [7., 0.],
                 [7., 4.],
                 [0., 4.]]])
    """
    dst_w = boxes[:, 1, 0] - boxes[:, 0, 0]
    dst_h = boxes[:, 3, 1] - boxes[:, 1, 1]
    zeros = torch.zeros_like(dst_w)

    points_dst: torch.Tensor = torch.stack([
        torch.stack([zeros, zeros], dim=-1),
        torch.stack([dst_w, zeros], dim=-1),
        torch.stack([dst_w, dst_h], dim=-1),
        torch.stack([zeros, dst_h], dim=-1)
    ], dim=1)
    return points_dst


def bbox_to_mask(boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert 2D bounding boxes to masks.

    Args:
        boxes (torch.Tensor): a tensor containing the coordinates of the bounding boxes to be extracted.
            The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise)
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in the x, y order.
        mask (torch.Tensor): the mask to apply cropping, which can be differentiable with `requires_grad=True`.

    Returns:
        torch.Tensor: the output mask tensor.

    Examples:
        >>> boxes = torch.tensor([[
        ...        [1., 1.],
        ...        [3., 1.],
        ...        [3., 2.],
        ...        [1., 2.],
        ...   ]])  # 1x4x2
        >>> mask =  torch.ones(1, 1, 5, 5, requires_grad=True)
        >>> out = bbox_to_mask(boxes, mask)
        >>> out
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]]], grad_fn=<CatBackward>)
        >>> kornia.utils.gradient_printer(out, torch.ones_like(mask), mask)
        tensor([[[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000, -0.0400, -0.0400, -0.0400,  0.0000],
                  [ 0.0000, -0.0400, -0.0400, -0.0400,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                  [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]])
    """
    validate_bboxes(boxes)
    assert boxes.size(0) == mask.size(0), \
        f"batch size must be same between boxes and masks. Got {boxes.size(0)} and {mask.size(0)}."

    dst_bbox = generate_dst_bbox_from_bbox(boxes)

    # compute the padding element-wisely
    left = boxes[:, 0, 0]
    top = boxes[:, 0, 1]
    right = mask.size(-1) - boxes[:, 1, 0] - 1
    bottom = mask.size(-2) - boxes[:, 2, 1] - 1

    padded = []
    for _src, _dst, _mask, _l, _t, _r, _b in zip(boxes, dst_bbox, mask, left, top, right, bottom):
        # assert False, (_src, _dst)
        try:
            _roi = sampling_by_boxes(_mask[None], _src[None], _dst[None], interpolation='bilinear', align_corners=True)
            _roi_padded = torch.nn.functional.pad(_roi, (int(_l), int(_r), int(_t), int(_b)))
        except RuntimeError as e:
            # Skip when cropped area is 0.
            # RuntimeError: solve_cpu: For batch 0: U(1,1) is zero, singular U.
            _roi_padded = _mask * 0.

        padded.append(_roi_padded)

    return torch.cat(padded)


def crop_by_boxes(
    tensor: torch.Tensor, src_box: torch.Tensor, src_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Perform crop transform on 2D images (4D tensor) by bounding boxes without manipulating tensor pixels.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Different from ``sampling_by_boxes``, this function differentiated on the cropping mask than input tensor.

    Args:
        tensor (torch.Tensor): the 2D image tensor with shape (B, C, H, W).
        src_box (torch.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        src_mask (torch.Tensor, optional): the mask to apply cropping on, which can be differentiable with
            `requires_grad=True`.

    Returns:
        torch.Tensor: the output tensor with patches.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32, requires_grad=True).reshape((1, 1, 4, 4)).repeat(2, 1, 1, 1)
        >>> src_box = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).repeat(2, 1, 1)  # 1x4x2
        >>> out = crop_by_boxes(input, src_box)
        >>> out
        tensor([[[ 5.,  6.],
                 [ 9., 10.]],
        <BLANKLINE>
                [[ 5.,  6.],
                 [ 9., 10.]]], grad_fn=<CatBackward>)
        >>> target = torch.ones(1, 1, 2, 2) * 8
        >>> _ = out.register_hook(lambda x: print(x))
        >>> loss = (target - out).mean()
        >>> loss.backward()
        tensor([[[-0.1250, -0.1250],
                 [-0.1250, -0.1250]],
        <BLANKLINE>
                [[-0.1250, -0.1250],
                 [-0.1250, -0.1250]]])
    """
    validate_bboxes(src_box)

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    b, c, h, w = tensor.shape

    if src_mask is None:
        src_mask = torch.ones(b, 1, h, w)

    mask = bbox_to_mask(src_box, src_mask)
    masked_tensor = tensor * mask

    left = src_box[:, 0, 0]
    top = src_box[:, 0, 1]
    right = w - src_box[:, 1, 0] - 1
    bottom = h - src_box[:, 2, 1] - 1

    cropped = []
    for _masked, _l, _t, _r, _b in zip(masked_tensor, left, top, right, bottom):
        cropped.append(_masked[..., int(_t):h - int(_b), int(_l):w - int(_r)])
    return torch.cat(cropped)
