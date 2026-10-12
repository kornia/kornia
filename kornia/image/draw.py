# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utilities for drawing on images."""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE

# TODO: implement width of the line


def _round_to_pixel(coords: Tensor) -> Tensor:
    """Return the int64 pixel whose centre is nearest to each coordinate; a coordinate exactly halfway rounds up.

    Pixel ``i`` covers ``[i - 0.5, i + 0.5)``, so ``-0.5`` lands on pixel 0 and ``2.5`` on pixel 3.
    """
    if coords.is_floating_point():
        # x + 0.5 is not representable in float16 / bfloat16 for larger integers (bfloat16 129 + 0.5 -> 130)
        coords = coords.to(torch.promote_types(coords.dtype, torch.float32))
        coords = torch.floor(coords + 0.5)
    return coords.to(torch.int64)


def _rectangle_bands(
    x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor, h: int, w: int, line_width: int, fill: bool
) -> Tensor:
    """Return the row and column slices ``[r0, r1, c0, c1]`` of each rectangle's bands, clipped to a ``h x w`` image.

    The corners are inclusive and ordered (``x1 <= x2``, ``y1 <= y2``) and share any leading shape. The outline is four
    bands ``line_width`` pixels thick, inside the box; a fill is one band. Each band is drawn as
    ``image[..., r0:r1, c0:c1]``. Its bounds are clamped to the image, so the part of a band outside the image is an
    empty slice: a one-pixel side outside the image disappears instead of being moved onto the border. Clamping instead
    of testing the bounds in Python keeps every corner a tensor until the one ``tolist`` in ``draw_rectangle``, so the
    function exports to ONNX.

    Returns:
        an int64 tensor of shape ``(*, K, 4)``: ``K`` is 1 for a fill and 4 for an outline.
    """
    if fill:
        bands = [(y1, y2, x1, x2)]
    else:
        bands = [
            (y1, torch.minimum(y1 + line_width - 1, y2), x1, x2),  # top
            (torch.maximum(y2 - line_width + 1, y1), y2, x1, x2),  # bottom
            (y1, y2, x1, torch.minimum(x1 + line_width - 1, x2)),  # left
            (y1, y2, torch.maximum(x2 - line_width + 1, x1), x2),  # right
        ]
    r0, r1, c0, c1 = (torch.stack(bound, -1) for bound in zip(*bands))
    # a start clamped to [0, size] and a stop to [0, size] give an empty slice whenever the band misses the image
    return torch.stack([r0.clamp(0, h), r1.clamp(-1, h - 1) + 1, c0.clamp(0, w), c1.clamp(-1, w - 1) + 1], -1)


def draw_point2d(image: Tensor, points: Tensor, color: Tensor) -> Tensor:
    r"""Set one or more coordinates in a Tensor to a color.

    Args:
        image: the input image on which to draw the points with shape :math`(C,H,W)` or :math`(H,W)`.
        points: the [x, y] points to be drawn on the image with shape :math`(N, 2)`, a single
            point with shape :math`(2,)`, or an empty tensor with shape :math`(0, 2)`.
        color: the color of the pixel with :math`(C)` where :math`C` is the number of channels of the image.
            A single value, with shape :math`(1)` or 0-d, is written to every channel.

    Return:
        The image with points set to the color. This operation modifies image inplace but also
        returns the drawn tensor for convenience. An empty point set leaves the image unchanged.

    Raises:
        ValueError: if the image is not ``(C, H, W)`` or ``(H, W)``, or a point lies outside the image after rounding.
        BaseError: if ``color`` has neither 1 nor ``C`` values on a ``(C, H, W)`` image, or ``points`` is not
            ``(2,)`` or ``(N, 2)``.

    Fractional coordinates go to the pixel whose centre is nearest; a coordinate exactly halfway rounds up.
    A point outside the image raises `ValueError`.

    """
    if image.dim() not in (2, 3):
        raise ValueError("image must have shape (C, H, W) or (H, W).")
    # A 0-d scalar has no channel dimension for the check below to read; treat it as one channel, as draw_line does.
    if color.ndim == 0:
        color = color.unsqueeze(0)
    # A single value is written to every channel (see the docstring).
    KORNIA_CHECK(
        (len(image.shape) == 2 and len(color.shape) == 1) or color.shape[0] in (1, image.shape[0]),
        "Color dim must match the channel dims of the provided image",
    )
    # round on the points' own device: MPS has no float64, and float64 points from NumPy are common
    points = _round_to_pixel(points).to(image.device)
    # A single [x, y] vector is a common call shape; zip(*points) iterated 0-d
    # scalars and raised TypeError. An empty (0, 2) set used to fail unpacking;
    # indexing with empty columns now leaves the image unchanged.
    if points.ndim == 1:
        KORNIA_CHECK(points.numel() == 2, "A 1D points tensor must have shape (2,) as [x, y]")
        points = points.unsqueeze(0)
    KORNIA_CHECK(points.ndim == 2 and points.shape[-1] == 2, "points must have shape (N, 2)")
    x = points[:, 0]
    y = points[:, 1]
    height, width = image.shape[-2:]
    if ((x < 0) | (x >= width) | (y < 0) | (y >= height)).any():
        raise ValueError(
            f"points must lie inside the {height}x{width} image after rounding to the nearest pixel centre: "
            f"x in [0, {width - 1}], y in [0, {height - 1}]."
        )
    if len(color.shape) == 1:
        color = torch.unsqueeze(color, dim=1)
    color = color.to(dtype=image.dtype, device=image.device)
    if len(image.shape) == 2:
        image[y, x] = color
    else:
        image[:, y, x] = color.expand(image.shape[0], -1)
    return image


def _draw_pixel(image: torch.Tensor, x: int, y: int, color: torch.Tensor) -> None:
    r"""Draw a pixel into an image.

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

    Each line is rasterized one pixel per step along its major axis, the one with the larger extent:
    at step ``t = 0 .. major`` the pixel is the one whose centre is nearest to the ideal segment, a tie
    rounding up in absolute coordinates, computed in integers. The line is therefore the same for ``p1 -> p2`` and
    ``p2 -> p1``, and both ``p1`` and ``p2`` are always drawn. Fractional endpoints go to the pixel whose centre is
    nearest; a coordinate exactly halfway rounds up.

    Args:
        image: the input image to where to draw the lines with shape :math`(C,H,W)`.
        p1: the start point [x y] of the line with shape (2, ) or (B, 2).
        p2: the end point [x y] of the line, with the same shape as ``p1``.
        color: the color of the line with shape :math`(C)` where :math`C` is the number of channels
            of the image. A single value, with shape :math`(1)` or 0-d, is written to every channel.
            Porting note: OpenCV's and Pillow's scalar colours fill only the first channel.

    Return:
        The image containing the line. This operation modifies image inplace but also returns
        the drawn tensor for convenience.

    Raises:
        ValueError: if ``p1`` and ``p2`` differ in shape or their last dimension is not 2, an endpoint lies outside the
            image after rounding, the image is not ``(C, H, W)``, or ``color`` has neither 1 nor ``C`` values.

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
    # equal shapes, not just equal leading sizes: a (2,) p1 would otherwise broadcast against a (2, 2) p2
    if (p1.shape != p2.shape) or (p1.shape[-1] != 2):
        raise ValueError(
            "Input points must be 2D points with shape (2, ) or (B, 2) and must have the same batch sizes."
        )
    # round on the points' own device: MPS has no float64, and float64 points from NumPy are common
    p1 = _round_to_pixel(p1).to(image.device)
    p2 = _round_to_pixel(p2).to(image.device)
    for name, p in (("p1", p1), ("p2", p2)):
        if ((p[..., 0] < 0) | (p[..., 0] >= image.shape[-1]) | (p[..., 1] < 0) | (p[..., 1] >= image.shape[-2])).any():
            raise ValueError(f"{name} is out of bounds.")

    if len(image.size()) != 3:
        raise ValueError("image must have 3 dimensions (C,H,W).")

    # A 0-d scalar (e.g. torch.tensor(255) for grayscale) is a common call shape;
    # color.size(0) used to IndexError. Treat it as a length-1 channel vector.
    if color.ndim == 0:
        color = color.unsqueeze(0)

    if color.size(0) not in (1, image.size(0)):
        raise ValueError("color must have the same number of channels as the image.")

    # move color to the same device and dtype as the input image
    color = color.to(image)

    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    x1, y1 = p1[:, 0:1], p1[:, 1:2]
    dx = p2[:, 0:1] - x1
    dy = p2[:, 1:2] - y1
    dx_sign = torch.sign(dx)
    dy_sign = torch.sign(dy)
    dx, dy = torch.abs(dx), torch.abs(dy)

    # Every line steps one pixel at a time along its major axis and places the minor coordinate at the pixel
    # nearest to m1 + s * t * minor / major, halfway rounding up in absolute coordinates (so p1 <-> p2 gives the
    # same pixels). The (line, t) pairs with t <= major are taken first, so the work is one entry per drawn
    # pixel rather than a (B, longest line) grid that pads every short line out. The rounding is taken in
    # integers: a float step lands just off an exact tie. A vertical or horizontal line has minor = 0, and a
    # single point has major = 0, which the clamp keeps from dividing by 0. An empty batch draws nothing.
    x_major = dx > dy
    major = torch.where(x_major, dx, dy)
    minor = torch.where(x_major, dy, dx)
    num_steps = int(major.max()) + 1 if major.numel() else 1
    steps = torch.arange(num_steps, device=image.device).unsqueeze(0)
    line, t = (steps <= major).nonzero(as_tuple=True)
    divisor = major[line, 0].clamp_min(1)
    x_maj = x_major[line, 0]
    m1 = torch.where(x_maj, y1[line, 0], x1[line, 0])
    s_minor = torch.where(x_maj, dy_sign[line, 0], dx_sign[line, 0])
    # floor((2 * m1 * major + 2 * s * t * minor + major) / (2 * major)) = round-half-up(m1 + s * t * minor / major)
    m_coords = torch.div(
        2 * m1 * divisor + 2 * s_minor * t * minor[line, 0] + divisor, 2 * divisor, rounding_mode="floor"
    )
    x_coords = torch.where(x_maj, x1[line, 0] + dx_sign[line, 0] * t, m_coords)
    y_coords = torch.where(x_maj, m_coords, y1[line, 0] + dy_sign[line, 0] * t)

    x_coords = torch.clamp(x_coords, min=0, max=image.shape[-1] - 1)
    y_coords = torch.clamp(y_coords, min=0, max=image.shape[-2] - 1)
    image[:, y_coords, x_coords] = color.view(-1, 1).expand(image.size(0), -1)
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
        color: a 0-d, size 1, size 3, BxNx1, or BxNx3 tensor.
            If C is 3, and color is 1 channel it will be broadcasted.
        fill: is a flag used to fill the boxes with color if True.

    Corners may come in either order. Coordinates are rounded to the nearest pixel centre (a coordinate exactly
    halfway rounds up), and the rectangle includes both corners. A side outside the image is not drawn and a fill is
    clipped to the image.

    Raises:
        ShapeError: if the image is not ``(B, C, H, W)``. It is also a ``ValueError``.
        ValueError: if ``rectangle`` is not ``(B, N, 4)``, or its batch size differs from the image's.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Example:
        >>> img = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
        >>> out = draw_rectangle(img, rect)

    """
    KORNIA_CHECK_SHAPE(image, ["B", "C", "H", "W"])
    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    if batch != batch_rect:
        raise ValueError("Image batch and rectangle batch must be equal")
    if num_points != 4:
        raise ValueError("Number of points in rectangle must be 4")

    rect = _round_to_pixel(rectangle)
    x1 = torch.minimum(rect[..., 0], rect[..., 2])
    x2 = torch.maximum(rect[..., 0], rect[..., 2])
    y1 = torch.minimum(rect[..., 1], rect[..., 3])
    y2 = torch.maximum(rect[..., 1], rect[..., 3])

    if color is None:
        color = torch.tensor([0.0] * c).expand(batch, num_rectangle, c)

    if fill is None:
        fill = False

    if len(color.shape) <= 1:
        color = color.expand(batch, num_rectangle, c)
    b, n, color_channels = color.shape

    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)

    # one host read for the bands of every rectangle
    bands = _rectangle_bands(x1, y1, x2, y2, h, w, 1, fill).tolist()
    for b in range(batch):
        for n in range(num_rectangle):
            for r0, r1, c0, c1 in bands[b][n]:
                image[b, :, r0:r1, c0:c1] = color[b, n, :, None, None]

    return image


def _get_convex_edges(polygon: Tensor, h: int, w: int) -> Tuple[Tensor, Tensor]:
    r"""Get the left and right edges of a polygon for each y-coordinate y \in [0, h).

    Args:
        polygon: represents polygons to draw in BxNx2
            N is the number of points
            2 is (x, y).
        h: bottom most coordinate (top coordinate is assumed to be 0)
        w: right most coordinate (left coordinate is assumed to be 0)

    Returns:
        The left and right edges of the polygon of shape (B,B).

    """
    # The edge maths runs in at least float32: half-precision vertices cannot resolve the scanline grid.
    dtype = torch.promote_types(polygon.dtype, torch.float32)
    polygon = polygon.to(dtype)

    # A single vertex is its own first and last point, so the loop below would leave no edge;
    # draw it as a zero-length edge, like a two-vertex polygon whose vertices coincide.
    if polygon.shape[-2] == 1:
        polygon = torch.cat((polygon, polygon), dim=-2)

    # Check if polygons are in loop closed format, if not -> make it so
    if not torch.allclose(polygon[..., -1, :], polygon[..., 0, :]):
        polygon = torch.cat((polygon, polygon[..., :1, :]), dim=-2)  # (B, N+1, 2)

    # Partition points into edges
    x_start, y_start = polygon[..., :-1, 0], polygon[..., :-1, 1]
    x_end, y_end = polygon[..., 1:, 0], polygon[..., 1:, 1]

    # Create scanlines, edge dx/dy, and produce x values
    ys = torch.arange(h, device=polygon.device, dtype=dtype)
    # A horizontal or zero-length edge is active only on its own scanline, where xs is x_start for any finite dx,
    # so its dx is set to 0. An epsilon added to dy instead underflows in float16 (0 / 0 blanks the scanline), and
    # in float64 it shifts a sloped edge enough to drop a pixel centre lying exactly on it.
    dy = y_end - y_start
    dx = x_end - x_start
    # Multiply before dividing: dx / dy first rounds the slope, which drops a centre lying exactly on the edge.
    safe_dy = torch.where(dy == 0, torch.ones_like(dy), dy)
    xs = torch.where(
        dy[..., None, :] == 0,
        x_start[..., None, :],
        (ys[..., :, None] - y_start[..., None, :]) * dx[..., None, :] / safe_dy[..., None, :] + x_start[..., None, :],
    )

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
    r"""Convert a List of variable length polygons into a fixed size tensor.

    Works by repeating the last element in the tensor.

    Args:
        polygons: List of variable length polygons of shape [N_1 x 2, N_2 x 2, ..., N_B x 2].
                    B is the batch size,
                    N_i is the number of points,
                    2 is (x, y).

    Returns:
        A fixed size tensor of shape (B, N, 2) where N = max_i(N_i)

    """
    B, N = len(polygons), len(max(polygons, key=len))
    batched_polygons = torch.zeros(B, N, 2, dtype=polygons[0].dtype, device=polygons[0].device)
    for b, p in enumerate(polygons):
        if len(p) == 0:
            # No last vertex to repeat; the row stays zero and the caller must not fill it.
            continue
        batched_polygons[b] = torch.cat((p, p[-1:].expand(N - len(p), 2))) if len(p) < N else p
    return batched_polygons


def draw_convex_polygon(images: Tensor, polygons: Union[Tensor, List[Tensor]], colors: Tensor) -> Tensor:
    r"""Draws convex polygons on a batch of image tensors.

    Args:
        images: is tensor of BxCxHxW.
        polygons: represents polygons as points, either BxNx2 or List of variable length polygons.
            N is the number of points.
            2 is (x, y).
        colors: a ``B x C`` tensor, a ``C`` tensor, or a 0-d scalar with the fill colour.

    Returns:
        This operation modifies image inplace but also returns the drawn tensor for
        convenience with same shape the of the input BxCxHxW.

    Raises:
        ShapeError: if the images are not ``(B, C, H, W)``. It is also a ``ValueError``.
        ValueError: if ``polygons`` is a tensor that is not three dimensional.
        BaseError: if the vertices are not ``(x, y)`` points, or the images, polygons and colors differ in batch size
            or device.

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
    empty_polygons = None
    if isinstance(polygons, List):
        empty_polygons = torch.tensor([len(p) == 0 for p in polygons], dtype=torch.bool, device=device)
        # `[]` has no polygon to take a dtype or device from; an empty (0, 0, 2) batch returns below.
        polygons = _batch_polygons(polygons) if polygons else images.new_zeros(0, 0, 2)
    b_p, _, xy, device_p = *polygons.shape, polygons.device
    if len(colors.shape) <= 1:
        colors = colors.expand(b_i, c_i)
    b_c, _, device_c = *colors.shape, colors.device
    KORNIA_CHECK(xy == 2, "Polygon vertices must be xy, i.e. 2-dimensional")
    KORNIA_CHECK(b_i == b_p == b_c, "Image, polygon, and color must have same batch dimension")
    KORNIA_CHECK(device == device_p == device_c, "Image, polygon, and color must have same device")
    # A polygon without vertices has nothing to fill, and closing its loop below needs a vertex.
    if polygons.shape[1] == 0:
        return images

    x_left, x_right = _get_convex_edges(polygons, h_i, w_i)
    ws = torch.arange(w_i, device=device, dtype=x_left.dtype)[None, None, :]
    fill_region = (ws >= x_left[..., :, None]) & (ws <= x_right[..., :, None])
    if empty_polygons is not None:
        fill_region &= ~empty_polygons[:, None, None]
    fill = fill_region[:, None]
    images.copy_(torch.where(fill, colors[..., None, None].to(images.dtype), images))
    return images
