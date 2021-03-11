from typing import Optional, Union

import torch


def draw_rectangle(
    image: torch.Tensor,
    rectangle: torch.Tensor,
    color: Optional[torch.Tensor] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    batch_id: Optional[List[int]] = None
) -> torch.Tensor:
"""Draws N rectangles on a batch of image tensors
    Args:
        image (torch.Tensor): is tensor of BxCxHxW.
        rectangle (torch.Tensor): represents number of rectangles to draw in Nx4
            N is the number of boxes to draw per batch [x1, y1, x2, y2]
            4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
        color (int or torch.Tensor, optional): can be a tensor. If image channels > 1
             color channel will be broadcasted if only 1. Default: None (black).
        fill (bool, optional): is a flag used to fill the boxes with color if True. Default: False.
        width (int): The line width. Default: 1. (Not implemented yet).
        batch_id (List[int], optional): A tensor corresponding to the indices of the image batch for each box.

    Returns:
        torch.Tensor: This operation modifies image inplace but also returns the drawn tensor for convenience with same shape the of the input BxCxHxW.

    Example:
        >>> img = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[0, 0, 4, 4], [4, 4, 10, 10]])
        >>> out = draw_rectangle(img, rect, batch_id=[0, 1])
"""
    # TODO
    # switch x and y dims
    # handle different color shapes
    # vectorise with boxes.long
    batch, c, h, w = image.shape
    _, num_rectangle, num_points = rectangle.shape

    # clone rectangle, in case it's been expanded assignment from clipping causes problems
    rectangle = rectangle.long()

    # clip rectangle to hxw bounds
    rectangle[:, :, ::2] = torch.clamp(rectangle[:, :, ::2], 0, h - 1)
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, w - 1)
    diff = rectangle[:, :, 2:] - rectangle[:, :, :2]
    # REMOVE
    #if torch.any(diff < 0):
    #    raise ValueError("All points must correspond to sequent top left followed by bottom right")

    if type(color) is float:
        color = torch.tensor([color])

    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                image[b, :, int(rectangle[b, n, 0]):int(rectangle[b, n, 2]) + 1,
                      int(rectangle[b, n, 1]):int(rectangle[b, n, 3]) + 1] = color[:, None, None]
            else:
                image[b, :, rectangle[b, n, 0]:rectangle[b, n, 2] + 1, rectangle[b, n, 1]] = color[:, None]
                image[b, :, rectangle[b, n, 0]:rectangle[b, n, 2] + 1, rectangle[b, n, 3]] = color[:, None]
                image[b, :, rectangle[b, n, 0], rectangle[b, n, 1]:rectangle[b, n, 3] + 1] = color[:, None]
                image[b, :, rectangle[b, n, 2], rectangle[b, n, 1]:rectangle[b, n, 3] + 1] = color[:, None]

    return image


def rectangle(
        image: torch.Tensor,
        boxes: torch.Tensor,
        color: Optional[Union[float, torch.Tensor]] = 1.0,
        fill: Optional[bool] = False) -> torch.Tensor:
    """Draws N rectangles on a batch of image tensors

    Args:
        image (torch.Tensor): is tensor of BxCxHxW
        boxes (torch.Tensor): represents number of rectangles to draw in BxNx4
            N is the number of boxes to draw per batch
            4 is in (top_left.y, top_left.x, bot_right.y, bot_right.x) ie h1, w1, h2, w2
        color (int or torch.Tensor): can be a float or tensor if image channels > 1
            if float is used and channels > 1, float will be broadcasted to all channels
        fill: bool is a flag used to fill the boxes with color if True

    Returns:
        torch.Tensor: This operation modifies image inplace but also returns the drawn tensor for convenience
    """

    batch, c, h, w = image.shape
    _, num_boxes, num_points = boxes.shape

    # clone boxes, in case it's been expanded assignment from clipping causes problems
    boxes = boxes.clone()

    # clip boxes to hxw bounds
    boxes[:, :, ::2] = torch.clamp(boxes[:, :, ::2], 0, h - 1)
    boxes[:, :, 1::2] = torch.clamp(boxes[:, :, 1::2], 0, w - 1)
    diff = boxes[:, :, 2:] - boxes[:, :, :2]
    if torch.any(diff < 0):
        raise ValueError("All points must correspond to sequent top left followed by bottom right")

    if type(color) is float:
        color = torch.tensor([color])

    for b in range(batch):
        for n in range(num_boxes):
            if fill:
                image[b, :, int(boxes[b, n, 0]):int(boxes[b, n, 2]) + 1,
                      int(boxes[b, n, 1]):int(boxes[b, n, 3]) + 1] = color[:, None, None]
            else:
                image[b, :, int(boxes[b, n, 0]):int(boxes[b, n, 2]) + 1, int(boxes[b, n, 1])] = color[:, None]
                image[b, :, int(boxes[b, n, 0]):int(boxes[b, n, 2]) + 1, int(boxes[b, n, 3])] = color[:, None]
                image[b, :, int(boxes[b, n, 0]), int(boxes[b, n, 1]):int(boxes[b, n, 3]) + 1] = color[:, None]
                image[b, :, int(boxes[b, n, 2]), int(boxes[b, n, 1]):int(boxes[b, n, 3]) + 1] = color[:, None]

    return image
