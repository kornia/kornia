from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE

# TODO: implement width of the line


def draw_point2d(image: Tensor, points: Tensor, color: Tensor) -> Tensor:
    r"""Sets one or more coordinates in a Tensor to a color.

    Args:
        image: the input image on which to draw the points with shape :math`(C,H,W)` or :math`(H,W)`.
        points: the [x, y] points to be drawn on the image.
        color: the color of the pixel with :math`(C)` where :math`C` is the number of channels of the image.

    Return:
        The image with points set to the color.
    """
    KORNIA_CHECK(
        (len(image.shape) == 2 and len(color.shape) == 1) or (image.shape[0] == color.shape[0]),
        "Color dim must match the channel dims of the provided image",
    )
    points = points.to(dtype=torch.int64, device=image.device)
    x, y = zip(*points)
    if len(color.shape) == 1:
        color = torch.unsqueeze(color, dim=1)
    color = color.to(dtype=image.dtype, device=image.device)
    if len(image.shape) == 2:
        image[y, x] = color
    else:
        image[:, y, x] = color
    return image


def _draw_pixel(image: torch.Tensor, x: int, y: int, color: torch.Tensor) -> None:
    r"""Draws a pixel into an image.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        x: the x coordinate of the pixel.
        y: the y coordinate of the pixel.
        color: the color of the pixel with :math`(C)` where :math`C` is the number of channels of the image.

    Return:
        Nothing is returned.
    """
    image[:, y, x] = color


def draw_line(image: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    r"""Draw a single line into an image.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        p1: the start point [x y] of the line with shape (2, ) or (B, 2).
        p2: the end point [x y] of the line with shape (2, ) or (B, 2).
        color: the color of the line with shape :math`(C)` where :math`C` is the number of channels of the image.
    Return:
        the image with containing the line.

    Examples:
        >>> image = torch.zeros(1, 8, 8)
        >>> draw_line(image, torch.tensor([6, 4]), torch.tensor([1, 4]), torch.tensor([255]))
        tensor([[[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0., 255., 255., 255., 255., 255., 255.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                 [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]])
    """
    if (p1.shape[0] != p2.shape[0]) or (p1.shape[-1] != 2 or p2.shape[-1] != 2):
        raise ValueError(
            "Input points must be 2D points with shape (2, ) or (B, 2) and must have the same batch sizes."
        )
    if (
        (p1[..., 0] < 0).any()
        or (p1[..., 0] >= image.shape[-1]).any()
        or (p1[..., 1] < 0).any()
        or (p1[..., 1] >= image.shape[-2]).any()
    ):
        raise ValueError("p1 is out of bounds.")
    if (
        (p2[..., 0] < 0).any()
        or (p2[..., 0] >= image.shape[-1]).any()
        or (p2[..., 1] < 0).any()
        or (p2[..., 1] >= image.shape[-2]).any()
    ):
        raise ValueError("p2 is out of bounds.")

    if len(image.size()) != 3:
        raise ValueError("image must have 3 dimensions (C,H,W).")

    if color.size(0) != image.size(0):
        raise ValueError("color must have the same number of channels as the image.")

    # move p1 and p2 to the same device as the input image
    # move color to the same device and dtype as the input image
    p1 = p1.to(image.device).to(torch.int64)
    p2 = p2.to(image.device).to(torch.int64)
    color = color.to(image)

    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    dx = x2 - x1
    dy = y2 - y1
    dx_sign = torch.sign(dx)
    dy_sign = torch.sign(dy)
    dx, dy = torch.abs(dx), torch.abs(dy)
    dx_zero_mask = dx == 0
    dy_zero_mask = dy == 0
    dx_gt_dy_mask = (dx > dy) & ~(dx_zero_mask | dy_zero_mask)
    rest_mask = ~(dx_zero_mask | dy_zero_mask | dx_gt_dy_mask)

    dx_zero_x_coords, dx_zero_y_coords = [], []
    dy_zero_x_coords, dy_zero_y_coords = [], []
    dx_gt_dy_x_coords, dx_gt_dy_y_coords = [], []
    rest_x_coords, rest_y_coords = [], []

    if dx_zero_mask.any():
        dx_zero_x_coords = [
            x for x_i, dy_i in zip(x1[dx_zero_mask], dy[dx_zero_mask]) for x in x_i.repeat(int(dy_i.item() + 1))
        ]
        dx_zero_y_coords = [
            y
            for y_i, s, dy_ in zip(y1[dx_zero_mask], dy_sign[dx_zero_mask], dy[dx_zero_mask])
            for y in (y_i + s * torch.arange(0, dy_ + 1, 1, device=image.device))
        ]

    if dy_zero_mask.any():
        dy_zero_x_coords = [
            x
            for x_i, s, dx_i in zip(x1[dy_zero_mask], dx_sign[dy_zero_mask], dx[dy_zero_mask])
            for x in (x_i + s * torch.arange(0, dx_i + 1, 1, device=image.device))
        ]
        dy_zero_y_coords = [
            y for y_i, dx_i in zip(y1[dy_zero_mask], dx[dy_zero_mask]) for y in y_i.repeat(int(dx_i.item() + 1))
        ]

    if dx_gt_dy_mask.any():
        dx_gt_dy_x_coords = [
            x
            for x_i, s, dx_i in zip(x1[dx_gt_dy_mask], dx_sign[dx_gt_dy_mask], dx[dx_gt_dy_mask])
            for x in (x_i + s * torch.arange(0, dx_i + 1, 1, device=image.device))
        ]
        dx_gt_dy_y_coords = [
            y
            for y_i, s, dx_i, dy_i in zip(
                y1[dx_gt_dy_mask], dy_sign[dx_gt_dy_mask], dx[dx_gt_dy_mask], dy[dx_gt_dy_mask]
            )
            for y in (
                y_i + s * torch.arange(0, dy_i + 1, dy_i / dx_i, device=image.device)[: int(dx_i.item()) + 1].ceil()
            )
        ]
    if rest_mask.any():
        rest_x_coords = [
            x
            for x_i, s, dx_i, dy_ in zip(x1[rest_mask], dx_sign[rest_mask], dx[rest_mask], dy[rest_mask])
            for x in (
                x_i + s * torch.arange(0, dx_i + 1, dx_i / dy_, device=image.device)[: int(dy_.item()) + 1].ceil()
            )
        ]
        rest_y_coords = [
            y
            for y_i, s, dy_i in zip(y1[rest_mask], dy_sign[rest_mask], dy[rest_mask])
            for y in (y_i + s * torch.arange(0, dy_i + 1, 1, device=image.device))
        ]
    x_coords = torch.tensor(dx_zero_x_coords + dy_zero_x_coords + dx_gt_dy_x_coords + rest_x_coords).long()
    y_coords = torch.tensor(dx_zero_y_coords + dy_zero_y_coords + dx_gt_dy_y_coords + rest_y_coords).long()
    image[:, y_coords, x_coords] = color.view(-1, 1)
    return image


def draw_rectangle(
    image: torch.Tensor, rectangle: torch.Tensor, color: Optional[torch.Tensor] = None, fill: Optional[bool] = None
) -> torch.Tensor:
    r"""Draw N rectangles on a batch of image tensors.

    Args:
        image: is tensor of BxCxHxW.
        rectangle: represents number of rectangles to draw in BxNx4
            N is the number of boxes to draw per batch index[x1, y1, x2, y2]
            4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
        color: a size 1, size 3, BxNx1, or BxNx3 tensor.
            If C is 3, and color is 1 channel it will be broadcasted.
        fill: is a flag used to fill the boxes with color if True.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Example:
        >>> img = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
        >>> out = draw_rectangle(img, rect)
    """

    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    if batch != batch_rect:
        raise AssertionError("Image batch and rectangle batch must be equal")
    if num_points != 4:
        raise AssertionError("Number of points in rectangle must be 4")

    # clone rectangle, in case it's been expanded assignment from clipping causes problems
    rectangle = rectangle.long().clone()

    # clip rectangle to hxw bounds
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, h - 1)
    rectangle[:, :, ::2] = torch.clamp(rectangle[:, :, ::2], 0, w - 1)

    if color is None:
        color = torch.tensor([0.0] * c).expand(batch, num_rectangle, c)

    if fill is None:
        fill = False

    if len(color.shape) == 1:
        color = color.expand(batch, num_rectangle, c)
    b, n, color_channels = color.shape

    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)

    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                image[
                    b,
                    :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1),
                ] = color[b, n, :, None, None]
            else:
                image[b, :, int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1), rectangle[b, n, 0]] = color[
                    b, n, :, None
                ]
                image[b, :, int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1), rectangle[b, n, 2]] = color[
                    b, n, :, None
                ]
                image[b, :, rectangle[b, n, 1], int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)] = color[
                    b, n, :, None
                ]
                image[b, :, rectangle[b, n, 3], int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)] = color[
                    b, n, :, None
                ]

    return image


def _get_convex_edges(polygon: Tensor, h: int, w: int) -> Tuple[Tensor, Tensor]:
    r"""Gets the left and right edges of a polygon for each y-coordinate y \in [0, h)
    Args:
        polygons: represents polygons to draw in BxNx2
            N is the number of points
            2 is (x, y).
        h: bottom most coordinate (top coordinate is assumed to be 0)
        w: right most coordinate (left coordinate is assumed to be 0)
    Returns:
        The left and right edges of the polygon of shape (B,B).
    """
    dtype = polygon.dtype

    # Check if polygons are in loop closed format, if not -> make it so
    if not torch.allclose(polygon[..., -1, :], polygon[..., 0, :]):
        polygon = torch.cat((polygon, polygon[..., :1, :]), dim=-2)  # (B, N+1, 2)

    # Partition points into edges
    x_start, y_start = polygon[..., :-1, 0], polygon[..., :-1, 1]
    x_end, y_end = polygon[..., 1:, 0], polygon[..., 1:, 1]

    # Create scanlines, edge dx/dy, and produce x values
    ys = torch.arange(h, device=polygon.device, dtype=dtype)
    dx = ((x_end - x_start) / (y_end - y_start + 1e-12)).clamp(-w, w)
    xs = (ys[..., :, None] - y_start[..., None, :]) * dx[..., None, :] + x_start[..., None, :]

    # Only count edge in their active regions (i.e between the vertices)
    valid_edges = (y_start[..., None, :] <= ys[..., :, None]).logical_and(ys[..., :, None] <= y_end[..., None, :])
    valid_edges |= (y_start[..., None, :] >= ys[..., :, None]).logical_and(ys[..., :, None] >= y_end[..., None, :])
    x_left_edges = xs.clone()
    x_left_edges[~valid_edges] = w
    x_right_edges = xs.clone()
    x_right_edges[~valid_edges] = -1

    # Find smallest and largest x values for the valid edges
    x_left = x_left_edges.min(dim=-1).values
    x_right = x_right_edges.max(dim=-1).values
    return x_left, x_right


def _batch_polygons(polygons: List[Tensor]) -> Tensor:
    r"""Converts a List of variable length polygons into a fixed size tensor.

    Works by repeating the last element in the tensor.
    Args:
        polygon: List of variable length polygons of shape [N_1 x 2, N_2 x 2, ..., N_B x 2].
                    B is the batch size,
                    N_i is the number of points,
                    2 is (x, y).
    Returns:
        A fixed size tensor of shape (B, N, 2) where N = max_i(N_i)
    """
    B, N = len(polygons), len(max(polygons, key=len))
    batched_polygons = torch.zeros(B, N, 2, dtype=polygons[0].dtype, device=polygons[0].device)
    for b, p in enumerate(polygons):
        batched_polygons[b] = torch.cat((p, p[-1:].expand(N - len(p), 2))) if len(p) < N else p
    return batched_polygons


def draw_convex_polygon(images: Tensor, polygons: Union[Tensor, List[Tensor]], colors: Tensor) -> Tensor:
    r"""Draws convex polygons on a batch of image tensors.

    Args:
        images: is tensor of BxCxHxW.
        polygons: represents polygons as points, either BxNx2 or List of variable length polygons.
            N is the number of points.
            2 is (x, y).
        color: a B x 3 tensor or 3 tensor with color to fill in.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Note:
        This function assumes a coordinate system (0, h - 1), (0, w - 1) in the image, with (0, 0) being the center
        of the top-left pixel and (w - 1, h - 1) being the center of the bottom-right coordinate.

    Example:
        >>> img = torch.rand(1, 3, 12, 16)
        >>> poly = torch.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]]])
        >>> color = torch.tensor([[0.5, 0.5, 0.5]])
        >>> out = draw_convex_polygon(img, poly, color)
    """
    # TODO: implement optional linetypes for smooth edges
    KORNIA_CHECK_SHAPE(images, ["B", "C", "H", "W"])
    b_i, c_i, h_i, w_i, device = *images.shape, images.device
    if isinstance(polygons, List):
        polygons = _batch_polygons(polygons)
    b_p, _, xy, device_p, dtype_p = *polygons.shape, polygons.device, polygons.dtype
    if len(colors.shape) == 1:
        colors = colors.expand(b_i, c_i)
    b_c, _, device_c = *colors.shape, colors.device
    KORNIA_CHECK(xy == 2, "Polygon vertices must be xy, i.e. 2-dimensional")
    KORNIA_CHECK(b_i == b_p == b_c, "Image, polygon, and color must have same batch dimension")
    KORNIA_CHECK(device == device_p == device_c, "Image, polygon, and color must have same device")

    x_left, x_right = _get_convex_edges(polygons, h_i, w_i)
    ws = torch.arange(w_i, device=device, dtype=dtype_p)[None, None, :]
    fill_region = (ws >= x_left[..., :, None]) & (ws <= x_right[..., :, None])
    images.mul_(~fill_region[:, None]).add_(fill_region[:, None] * colors[..., None, None])
    return images
