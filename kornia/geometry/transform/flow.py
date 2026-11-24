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

"""Warp dense optical flow by transforming both ends of each correspondence.

Use :func:`warp_optical_flow` with the source-to-destination matrix shared by the two images.
"""

from __future__ import annotations

import torch

from kornia.core._small_linalg import _adjugate_3x3
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.core.utils import _at_least_float32, _autocast_disabled
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.transform.imgwarp import _unit_projective_scale

__all__ = ["warp_optical_flow"]


def _sample_flow(flow: torch.Tensor, valid: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample in pixels, with border replication and exact validity at integer positions."""
    _, _, height, width = flow.shape
    x = xy[..., 0].clamp(0, width - 1)
    y = xy[..., 1].clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    fx, fy = x - x0, y - y0
    top, bottom = y0 * width, y1 * width
    indices = torch.stack((top + x0, top + x1, bottom + x0, bottom + x1), dim=-1)
    flat_indices = indices.flatten(1)
    neighbors = flow.flatten(2).gather(2, flat_indices[:, None].expand(-1, 2, -1)).unflatten(-1, (-1, 4))
    top_left, top_right, bottom_left, bottom_right = neighbors.unbind(dim=-1)
    known = valid.flatten(1).gather(1, flat_indices).reshape_as(indices)
    known_tl, known_tr, known_bl, known_br = known.unbind(dim=-1)
    # In each axis an exact integer uses only the lower index, independent of the neighboring pixel.
    no_x, no_y = fx == 0, fy == 0
    sample_valid = (known_tl & (no_x | known_tr) & (no_y | known_bl) & (no_x | no_y | known_br))[..., None]

    upper = top_left.lerp(top_right, fx[:, None])
    lower = bottom_left.lerp(bottom_right, fx[:, None])
    sample = upper.lerp(lower, fy[:, None]).transpose(1, 2)
    return torch.where(sample_valid, sample, 0), sample_valid


def warp_optical_flow(
    flow: torch.Tensor,
    transform: torch.Tensor,
    dsize: tuple[int, int],
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Warp dense optical flow when the same pixel homography transforms both frames.

    For each destination pixel centre :math:`q`, back-project :math:`p = H^{-1}(q)`, then bilinearly sample the
    source flow :math:`u` at :math:`p`. The output is the displacement :math:`H(p + u) - q` in destination pixels.
    Use the same source-to-destination :math:`H` for the two images with :func:`warp_perspective`.

    Convention:
        - Channel 0 is horizontal displacement in columns, and channel 1 is vertical displacement in rows.
          Pixel centres are integers. Each input batch item has its own homography.
        - A source position inside the footprint, :math:`[-0.5, W-0.5) \times [-0.5, H-0.5)`, samples with
          bilinear interpolation and border replication. All neighbors with nonzero weights must be valid.
          Nonfinite input vectors are invalid. Use the mask for finite unknown-value sentinels in datasets.
        - The output mask describes whether a source vector was sampled and both ends project to finite points.
          The mapped endpoint can be outside the destination canvas, as in a cropped image pair.
          Invalid vectors are returned as zero. A singular homography makes the whole sample invalid.
        - Flow and homography use the same device and floating dtype, or a float32 homography with float16 or
          bfloat16 flow. The half-precision paths compute in float32 and cast the vectors back. Keep the
          homography in float32 when a translation such as ``width - 1`` does not fit exactly in the flow dtype.

    Args:
        flow: Source flow tensor of shape ``(B, 2, H, W)``.
        transform: Finite source-to-destination pixel homographies of shape ``(B, 3, 3)``.
        dsize: Positive output height and width.
        valid: Optional boolean source validity mask of shape ``(B, 1, H, W)``.

    Returns:
        Destination vectors of shape ``(B, 2, h, w)`` and a boolean validity mask of shape ``(B, 1, h, w)``.

    See Also:
        - :func:`warp_perspective`: warp each image in the pair with the same source-to-destination homography.
        - :func:`~kornia.metrics.aepe`: compare flow vectors by endpoint error.

    """
    KORNIA_CHECK_SHAPE(flow, ["B", "2", "H", "W"])
    KORNIA_CHECK_SHAPE(transform, ["B", "3", "3"])
    KORNIA_CHECK(transform.shape[0] == flow.shape[0], "flow and transform must have the same batch size.")
    half = flow.dtype in (torch.float16, torch.bfloat16)
    KORNIA_CHECK(half or flow.dtype in (torch.float32, torch.float64), "flow must have a supported floating dtype.")
    KORNIA_CHECK(
        transform.dtype == flow.dtype or (half and transform.dtype == torch.float32),
        "transform must match the flow dtype or be float32 for half-precision flow.",
    )
    KORNIA_CHECK(flow.device == transform.device, "flow and transform must be on the same device.")
    height, width = flow.shape[-2:]
    out_height, out_width = dsize
    KORNIA_CHECK(height > 0 and width > 0, "flow must have positive spatial dimensions.")
    KORNIA_CHECK(out_height > 0 and out_width > 0, "dsize must contain a positive height and width.")
    known = torch.isfinite(flow).all(dim=1, keepdim=True)
    if valid is not None:
        KORNIA_CHECK_SHAPE(valid, ["B", "1", "H", "W"])
        KORNIA_CHECK(valid.shape == known.shape, "valid must match the batch and spatial dimensions of flow.")
        KORNIA_CHECK(
            valid.dtype == torch.bool and valid.device == flow.device, "valid must be boolean and on the flow device."
        )
        known = known & valid

    with _autocast_disabled(flow.device.type):
        vectors = torch.where(known, _at_least_float32(flow), 0)
        matrix = _at_least_float32(transform)
        matrix = matrix / _unit_projective_scale(matrix)
        # The adjugate is a projective inverse. Skip dividing the whole matrix by its determinant.
        inverse, determinant = _adjugate_3x3(matrix)
        dst = create_meshgrid(out_height, out_width, False, device=flow.device, dtype=matrix.dtype).flatten(1, 2)
        source_h = dst @ inverse[:, :, :2].mT + inverse[:, None, :, 2]
        w = source_h[..., 2:]
        xy = torch.where(w < 0, -source_h[..., :2], source_h[..., :2])
        w = w.abs()
        # Test the source bounds before division, so exact horizon points never enter the sampler.
        limit = xy.new_ones(2)
        limit = torch.stack((limit[0] * width, limit[1] * height)) - 0.5
        inside = (
            (determinant[:, None, None] != 0)
            & (w != 0)
            & ((xy >= -0.5 * w) & (xy < limit * w)).all(dim=-1, keepdim=True)
        )
        source = torch.where(inside, xy, 0) / torch.where(inside, w, 1)
        sampled, sampled_valid = _sample_flow(vectors, known, source)

        # (A - q c^T) u / (c^T (p + u) + d), for H = [A b; c^T d].
        # This avoids subtracting two nearly equal pixels when the flow is small.
        delta = sampled @ matrix[:, :, :2].mT
        endpoint_w = (source + sampled) @ matrix[:, 2:3, :2].mT + matrix[:, None, 2:3, 2]
        out_valid = inside & sampled_valid & (endpoint_w != 0)
        numerator = delta[..., :2] - dst * delta[..., 2:]
        result = (torch.where(out_valid, numerator, 0) / torch.where(out_valid, endpoint_w, 1)).to(flow.dtype)
        out_valid = out_valid & torch.isfinite(result).all(dim=-1, keepdim=True)
        result = torch.where(out_valid, result, 0)
    return (
        result.transpose(1, 2).reshape(flow.shape[0], 2, out_height, out_width),
        out_valid.transpose(1, 2).reshape(flow.shape[0], 1, out_height, out_width),
    )
