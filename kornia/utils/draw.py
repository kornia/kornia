from typing import Optional, Union

import torch


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
