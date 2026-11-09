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

from __future__ import annotations

import math

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_TYPE


def _use_eager_scratch(tensor: torch.Tensor) -> bool:
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type in ("cpu", "cuda", "mps")
        and not torch.jit.is_tracing()
        and not torch.compiler.is_compiling()
        and not torch._C._functorch.is_functorch_wrapped_tensor(tensor)
        and torch._C._len_torch_dispatch_stack() == 0
    )


def _needs_grad(tensor: torch.Tensor) -> bool:
    return (
        tensor.requires_grad
        or torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None
        or torch.jit.is_tracing()
        or torch._C._len_torch_dispatch_stack() > 0
    )


def _squared_distance_transform(cost: torch.Tensor) -> torch.Tensor:
    # Separable min-convolution with a parabola: Felzenszwalb & Huttenlocher (2012), section 2.2.
    # Work is O(numel * sum(spatial_shape)); tensor reductions avoid the data-dependent envelope stack.
    # Eager execution reuses a single scratch buffer across tiles and axes: 2**21 elements (8 MiB in
    # float32) on CPU with 64 queries keeps inputs in L2 and candidates in L3 below glibc's mmap threshold;
    # CUDA uses 2**23 (32 MiB) and other backends/compilers use 2**24 (64 MiB) with up to 1024 queries.
    # Inference uses SIMD-friendly amin without int64 index tracking; autograd detaches the tiled argmin
    # search and attaches gradients once per axis via gather (O(N) scatter_add backward instead of O(N*L)).
    use_scratch = _use_eager_scratch(cost)
    need_grad = _needs_grad(cost)
    dev_type = cost.device.type
    if torch.compiler.is_compiling():
        budget = 2**24
        max_queries = 1024
    elif dev_type == "cpu":
        budget = 2**21
        max_queries = 64
    elif dev_type == "cuda":
        budget = 2**23
        max_queries = 1024
    else:
        budget = 2**24
        max_queries = 1024

    buf: torch.Tensor | None = None
    for axis in range(2, cost.ndim):
        # Contiguous source lines avoid strided reductions, especially on CPU.
        values = cost.movedim(axis, -1).contiguous()
        length = values.shape[-1]
        positions = torch.arange(length, device=cost.device, dtype=cost.dtype)
        lines = values.reshape(-1, length)
        num_lines = lines.shape[0]
        query_tile = min(max_queries, length, max(1, budget // length))
        line_tile = max(1, budget // (query_tile * length))
        b_cap = min(num_lines, line_tile)
        if not need_grad:
            if use_scratch:
                need_buf = b_cap * query_tile * length
                if buf is None or buf.numel() < need_buf:
                    buf = cost.new_empty(need_buf)
                transformed = []
                for start in range(0, length, query_tile):
                    offsets = (positions[start : start + query_tile, None] - positions[None, :]).square()
                    q_len = offsets.shape[0]
                    col_out = lines.new_empty((num_lines, q_len))
                    for b_start in range(0, num_lines, line_tile):
                        b_end = min(b_start + line_tile, num_lines)
                        b_len = b_end - b_start
                        cur_buf = buf[: b_len * q_len * length].view(b_len, q_len, length)
                        torch.add(lines[b_start:b_end].unsqueeze(-2), offsets, out=cur_buf)
                        torch.amin(cur_buf, dim=-1, out=col_out[b_start:b_end])
                    transformed.append(col_out)
            else:
                transformed = []
                for start in range(0, length, query_tile):
                    offsets = (positions[start : start + query_tile, None] - positions[None, :]).square()
                    chunks = [(block.unsqueeze(-2) + offsets).amin(dim=-1) for block in lines.split(line_tile, dim=0)]
                    transformed.append(chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0))
            out_lines = transformed[0] if len(transformed) == 1 else torch.cat(transformed, dim=-1)
        else:
            lines_det = torch.autograd.forward_ad.unpack_dual(lines).primal.detach()
            if use_scratch:
                need_buf = b_cap * query_tile * length
                if buf is None or buf.numel() < need_buf:
                    buf = cost.new_empty(need_buf)
                val_discard = lines_det.new_empty((b_cap, query_tile))
                idx_tiles = []
                for start in range(0, length, query_tile):
                    offsets = (positions[start : start + query_tile, None] - positions[None, :]).square()
                    q_len = offsets.shape[0]
                    col_idx = torch.empty((num_lines, q_len), dtype=torch.int64, device=cost.device)
                    for b_start in range(0, num_lines, line_tile):
                        b_end = min(b_start + line_tile, num_lines)
                        b_len = b_end - b_start
                        cur_buf = buf[: b_len * q_len * length].view(b_len, q_len, length)
                        torch.add(lines_det[b_start:b_end].unsqueeze(-2), offsets, out=cur_buf)
                        torch.min(cur_buf, dim=-1, out=(val_discard[:b_len, :q_len], col_idx[b_start:b_end]))
                    idx_tiles.append(col_idx)
            else:
                idx_tiles = []
                for start in range(0, length, query_tile):
                    offsets = (positions[start : start + query_tile, None] - positions[None, :]).square()
                    chunks = [
                        (block.unsqueeze(-2) + offsets).min(dim=-1).indices
                        for block in lines_det.split(line_tile, dim=0)
                    ]
                    idx_tiles.append(chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0))
            idx = idx_tiles[0] if len(idx_tiles) == 1 else torch.cat(idx_tiles, dim=-1)
            out_lines = lines.gather(-1, idx) + (positions - idx.to(cost.dtype)).square()
        cost = out_lines.reshape(values.shape).movedim(-1, axis)
    return cost


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    r"""Compute Euclidean distances with differentiable source costs in 2D or 3D.

    ``distance_transform(image)`` maps each channel of ``(B, C, H, W)`` images or
    ``(B, C, D, H, W)`` volumes independently to distances in pixels. For binary masks,
    this is the Euclidean distance to the nearest positive pixel, up to floating-point
    rounding. Fractional masks use the generalized squared distance transform:

    .. math::
        D(p) = \sqrt{\min_{q: v(q)>0} \left(\|p-q\|_2^2 - h\log v(q)\right)},
        \qquad v = \operatorname{clip}(\mathrm{image}, 0, 1).

    The separable minimum-convolution formulation follows section 2.2 of
    `Distance Transforms of Sampled Functions (Felzenszwalb and Huttenlocher, 2012)
    <https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf>`_. The implementation uses
    tiled tensor reductions, not the paper's linear-time envelope algorithm: its work
    scales as the number of input elements times the sum of the spatial dimensions.

    Convention:
        - Distances are to positive pixels, whereas SciPy and OpenCV measure distances to
          zero pixels. On a nonempty binary mask this matches
          ``scipy.ndimage.distance_transform_edt(image == 0)`` per batch/channel.
        - A channel without positive pixels returns zeros, preserving the empty-mask convention.
        - A fractional source adds ``-h * log(v)`` to the **squared** distance, including at
          the source itself. Its influence extends throughout the image. Multiple sources
          compete by their minimum cost; they do not lower distances by being counted repeatedly.
        - The operation is piecewise differentiable. For a unique minimizing source and
          ``D(p) > 0``, its derivative is ``-h / (2 * v(q) * D(p))``. At ties, one minimizing
          source receives the gradient (the first selected by the axis-wise reductions).
          At zero distance the derivative is defined as zero to avoid the square-root singularity.
          Nonpositive inputs and values above 1 have zero gradient; the derivative of clipping
          at exactly 1 passes through. There is no straight-through estimator.
        - For dense fractional masks, a pixel with ``-h * log(v) < 1`` selects itself: every
          other pixel is at least one squared pixel away. Increase ``h`` to let stronger
          neighbours compete over longer distances.

    Args:
        image: Floating-point image or volume of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
        kernel_size: Odd integer of at least 3, retained for call compatibility. It does not
            change the result: the transform considers the entire spatial domain.
        h: Positive finite weight of the fractional-source cost in squared pixel units.
            It does not affect binary distances. Half-precision inputs are computed in float32
            and cast back.
            ``h``, source costs and gradients must remain representable in the working dtype.

    Returns:
        Distances with the input's shape, dtype and device.

    See Also:
        - :class:`DistanceTransform`: Module form of this operation.

    Example:
        >>> image = torch.zeros(1, 1, 5, 5)
        >>> image[:, :, 0, 0] = 1
        >>> distance_transform(image)[0, 0, 3, 4]
        tensor(5.)

    """
    KORNIA_CHECK_IS_TENSOR(image)
    KORNIA_CHECK(image.is_floating_point(), "image must be a floating point tensor")
    KORNIA_CHECK(image.ndim in (4, 5), "Invalid image shape, expected BxCxHxW or BxCxDxHxW")
    KORNIA_CHECK(all(size > 0 for size in image.shape[2:]), "Spatial dimensions must be nonempty")
    KORNIA_CHECK_TYPE(kernel_size, int, "kernel_size must be an int")
    KORNIA_CHECK(kernel_size % 2 != 0 and kernel_size >= 3, "kernel_size must be an odd integer >= 3")
    KORNIA_CHECK(math.isfinite(h) and h > 0, "h must be positive and finite")

    values = image.float() if image.dtype in (torch.float16, torch.bfloat16) else image
    if _use_eager_scratch(values) and not _needs_grad(values):
        cost = values.clamp(0.0, 1.0).log_().mul_(-h)
        squared = _squared_distance_transform(cost)
        squared.masked_fill_(squared == float("inf"), 0.0).sqrt_()
        return squared.to(image.dtype)

    # Use where at the clipping bound: clamp's derivative at 1 differs across supported Torch versions.
    # Substitute before log, not after: log(0) would poison backward even in an unselected where arm.
    one = values.new_ones(())
    zero = values.new_zeros(())
    inf = values.new_full((), float("inf"))
    positive = values > 0
    safe_values = torch.where(positive & (values <= 1), values, one)
    cost = torch.where(positive, -h * safe_values.log(), inf)
    squared = _squared_distance_transform(cost)
    valid = torch.isfinite(squared) & (squared > 0)
    # Likewise avoid differentiating sqrt(0), including on Torch 2.5 and half-precision inputs.
    safe_squared = torch.where(valid, squared, one)
    return torch.where(valid, safe_squared.sqrt(), zero).to(image.dtype)


class DistanceTransform(nn.Module):
    r"""Compute Euclidean distances with differentiable source costs in 2D or 3D.

    ``DistanceTransform(h=0.35)(image)`` applies :func:`distance_transform` independently
    to each channel of an image or volume, preserving its shape, dtype and device.
    Binary masks give Euclidean distances; fractional masks use differentiable source costs.

    Args:
        kernel_size: Odd integer of at least 3, retained for compatibility; does not affect distances.
        h: Positive finite weight of the squared-distance source cost. See :func:`distance_transform`.

    See Also:
        - :func:`distance_transform`: Functional form and gradient conventions.

    """

    def __init__(self, kernel_size: int = 3, h: float = 0.35) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute distances for a batch of images or volumes, independently per channel."""
        return distance_transform(image, self.kernel_size, self.h)
