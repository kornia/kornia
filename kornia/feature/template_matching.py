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

"""Batched zero-mean normalized cross-correlation for template localization."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

__all__ = ["match_template_zncc"]


def _center_patch(input: Tensor) -> tuple[Tensor, Tensor]:
    # One scale across channels preserves their relative contribution. Only scale extreme
    # magnitudes: subtracting a local anchor first preserves weak texture on a large DC offset.
    magnitude = input.detach().abs().amax(dim=(1, -2, -1), keepdim=True)
    limit = math.sqrt(torch.finfo(input.dtype).max / (256 * input.shape[1] * input.shape[-2] * input.shape[-1]))
    small = math.sqrt(torch.finfo(input.dtype).tiny) * 16
    scale = torch.where((magnitude > limit) | ((magnitude > 0) & (magnitude < small)), magnitude, 1.0)
    scaled = input / scale
    shifted = scaled - scaled[..., :1, :1]
    return shifted - shifted.mean(dim=(-2, -1), keepdim=True), scale


def _score_tile(image: Tensor, template: Tensor, min_variance: float) -> tuple[Tensor, Tensor]:
    patches = image.unfold(2, template.shape[-2], 1).unfold(3, template.shape[-1], 1)
    template = template[:, :, None, None]
    finite_p, finite_t = torch.isfinite(patches), torch.isfinite(template)
    p, pscale = _center_patch(torch.where(finite_p, patches, 0.0))
    t, tscale = _center_patch(torch.where(finite_t, template, 0.0))
    dims = (1, -2, -1)
    pvar, tvar = p.square().mean(dims, keepdim=True), t.square().mean(dims, keepdim=True)
    valid = finite_p.all(dim=dims, keepdim=True) & finite_t.all(dim=dims, keepdim=True)
    valid = valid & (pvar > (math.sqrt(min_variance) / pscale).square())
    valid = valid & (tvar > (math.sqrt(min_variance) / tscale).square())
    # Guard the operands of sqrt and division, including constant patches at min_variance=0.
    safe_p = torch.where(valid, pvar, 1.0)
    safe_t = torch.where(valid, tvar, 1.0)
    numerator = torch.where(valid, (p * t).mean(dims, keepdim=True), 0.0)
    scores = (numerator / safe_p.sqrt() / safe_t.sqrt()).clamp(-1.0, 1.0)
    return scores[..., 0, 0], valid[..., 0, 0]


def match_template_zncc(image: Tensor, template: Tensor, min_variance: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Match a template with per-channel zero-mean normalized cross-correlation (ZNCC).

    The score is the inner product of centered image and template windows, divided by their
    Euclidean norms. Each channel is centered spatially, then all channels are aggregated.
    This is the CCOEFF_NORMED formula documented in OpenCV's TemplateMatchModes:
    https://docs.opencv.org/4.13.0/df/dfb/group__imgproc__object.html.
    The classical normalized-correlation formulation is also described in :cite:`Lewis1995template`.
    Unlike OpenCV's constant-template convention, degenerate windows return zero and false.

    Args:
        image: Image batch of shape (B,C,H,W), in float32 or float64.
        template: Template of shape (B,C,h,w), or (1,C,h,w) shared across the batch.
            Its dtype and device must match image, and it must fit entirely inside the image.
        min_variance: Nonnegative finite threshold in squared intensity units. The average
            centered energy over all channels and spatial samples must exceed this threshold
            for both the template and window.

    Returns:
        Scores and a boolean numerical-validity mask, both of shape (B,1,H-h+1,W-w+1).
        Position (y,x) corresponds to the template's top-left corner. Valid scores lie in [-1,1],
        with larger scores indicating a better match. Invalid windows have score zero.

    Note:
        No padding, resizing or intensity rescaling is applied to the matching definition.
        A nonfinite image pixel invalidates only overlapping windows; a nonfinite template
        invalidates its image pair. The mask is not a confidence or visibility estimate.
        Mask invalid scores before selecting a peak and handle images with no valid window.
        The score map is differentiable on nondegenerate inputs; integer argmax locations are not.

        Per-channel intensity offsets and a common positive gain cancel in the ideal score.
        Independent per-channel gains need not cancel, and scaling can change the validity
        decision because min_variance is expressed in squared intensity units.

        Locally centered windows avoid cancellation in raw second moments on weak textures.
        Computation uses bounded spatial tiles rather than one materialized full patch tensor;
        differentiable tiles are recomputed during backward to avoid retaining all centered
        windows. This prioritizes numerical accuracy and bounded intermediates over the speed
        of a convolution-only implementation. Fixed-shape compilation and gradients on
        nondegenerate inputs are supported; dynamic-shape export is not guaranteed. This is
        direct spatial correlation, not the FFT or summed-area acceleration in the reference.
    """
    for name, value in (("image", image), ("template", template)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor.")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64.")
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W).")
    if image.dtype != template.dtype or image.device != template.device:
        raise ValueError("image and template must have the same dtype and device.")
    batch, channels, height, width = image.shape
    tb, tc, th, tw = template.shape
    if min(batch, channels, height, width, tb, tc, th, tw) < 1:
        raise ValueError("image and template dimensions must be nonempty.")
    if tb not in (1, batch) or tc != channels or th > height or tw > width:
        raise ValueError("template must have batch 1 or B, matching channels, and fit inside image.")
    if not math.isfinite(min_variance) or min_variance < 0:
        raise ValueError("min_variance must be finite and nonnegative.")

    # Crop before unfolding so backward never creates a full-image unfolded gradient.
    # extract_tensor_patches materializes all windows, which is deliberately avoided here.
    out_h, out_w = height - th + 1, width - tw + 1
    max_windows = max(1, 262144 // (batch * channels * th * tw))
    tile_w = min(out_w, max_windows)
    tile_h = max(1, max_windows // tile_w)
    rows, mask_rows = [], []
    for y in range(0, out_h, tile_h):
        cols, masks = [], []
        for x in range(0, out_w, tile_w):
            tile = image[:, :, y : y + tile_h + th - 1, x : x + tile_w + tw - 1]
            if torch.is_grad_enabled() and (image.requires_grad or template.requires_grad):
                scores, valid = checkpoint(
                    _score_tile, tile, template, min_variance, use_reentrant=False, preserve_rng_state=False
                )
            else:
                scores, valid = _score_tile(tile, template, min_variance)
            cols.append(scores)
            masks.append(valid)
        rows.append(torch.cat(cols, dim=-1))
        mask_rows.append(torch.cat(masks, dim=-1))
    return torch.cat(rows, dim=-2), torch.cat(mask_rows, dim=-2)
