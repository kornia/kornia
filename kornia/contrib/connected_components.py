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

import torch
import torch.nn.functional as F


def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Compute the Connected-component labelling (CCL) algorithm.

    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png

    The implementation is an adaptation of the following repository:

    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc

    .. warning::
        This is an experimental API subject to changes and optimization improvements.

    .. note::
       See a working example `here <https://www.kornia.org/tutorials/nbs/connected_components.html>`__.

    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.

    Return:
        The labels image with the same shape of the input image.

    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    H, W = image.shape[-2:]
    image_view = image.view(-1, 1, H, W)

    # precompute a mask with the valid values
    mask = image_view == 1

    # allocate the output tensors for labels
    B, _, _, _ = image_view.shape
    out = torch.arange(1, B * H * W + 1, device=image.device, dtype=torch.float32).view((-1, 1, H, W))
    out[~mask] = 0

    for _ in range(num_iterations):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication

    return out.to(image.dtype).view_as(image)


def connected_components_union_find(image: torch.Tensor) -> torch.Tensor:
    r"""Label 8-connected foreground components exactly using block union-find.

    This pure PyTorch implementation contracts each 2x2 block, following the
    block-based approach of Allegretti, Bolelli and Grana, *Optimized Block-Based
    Algorithms to Label Connected Components on GPUs*, TPDS 2020,
    https://doi.org/10.1109/TPDS.2019.2934683. Root hooking with ``scatter_reduce_``
    and pointer jumping replace the paper's custom CUDA union kernels.

    Unlike :func:`connected_components`, this function runs until convergence
    and returns integer labels, including for half-precision inputs. There is
    no iteration budget to choose based on component diameter.

    Args:
        image: Binary image of shape :math:`(*, 1, H, W)`. Boolean, integer and
            floating-point tensors are accepted. Only values equal to 1 are
            foreground, as in :func:`connected_components`.

    Returns:
        A ``torch.int64`` tensor on the input device with the same shape.
        Background is zero. Foreground labels are the minimum 1-based block
        index in each component, in raster order over flattened batches and
        the :math:`\lceil H/2\rceil \times \lceil W/2\rceil` block grid.
        Labels are deterministic and distinct across images, but not contiguous
        and not equal to the labels of the pooling implementation.

    Note:
        This discrete operation is not differentiable. Working memory is linear
        in the number of pixels. Convergence checks synchronize the device with
        the host; CUDA graph capture and ``torch.compile(fullgraph=True)`` are
        not supported. All mask, edge and label computation stays on the input
        device. The existing pooling API remains useful for fixed-work graphs.

    Example:
        >>> mask = torch.tensor([[[1, 0, 0], [0, 0, 1]]], dtype=torch.bool)
        >>> connected_components_union_find(mask)
        tensor([[[1, 0, 0],
                 [0, 0, 2]]])

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image is not a torch.Tensor. Got: {type(image)}")
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")
    if image.numel() == 0:
        return torch.zeros_like(image, dtype=torch.int64)

    height, width = image.shape[-2:]
    mask = (image == 1).reshape(-1, height, width)
    # MPSGraph's boolean padding can abort the process; allocate the border explicitly.
    padded = mask.new_zeros((mask.shape[0], height + height % 2, width + width % 2))
    padded[:, :height, :width] = mask
    top_left, top_right = padded[:, ::2, ::2], padded[:, ::2, 1::2]
    bottom_left, bottom_right = padded[:, 1::2, ::2], padded[:, 1::2, 1::2]
    # Every pair of foreground pixels inside a 2x2 block is 8-connected.
    # A block may connect to four already-visited neighbors, but only if an
    # actual pair of foreground pixels touches across the shared boundary.
    blocks = torch.arange(top_left.numel(), device=image.device).reshape(top_left.shape)
    left = (top_left[:, :, 1:] | bottom_left[:, :, 1:]) & (top_right[:, :, :-1] | bottom_right[:, :, :-1])
    above = (top_left[:, 1:, :] | top_right[:, 1:, :]) & (bottom_left[:, :-1, :] | bottom_right[:, :-1, :])
    above_left = top_left[:, 1:, 1:] & bottom_right[:, :-1, :-1]
    above_right = top_right[:, 1:, :-1] & bottom_left[:, :-1, 1:]

    sources = torch.cat(
        [
            blocks[:, :, 1:].flatten(),
            blocks[:, 1:, :].flatten(),
            blocks[:, 1:, 1:].flatten(),
            blocks[:, 1:, :-1].flatten(),
        ]
    )
    # Absent edges become self-loops. Fixed-size arrays avoid nonzero/boolean
    # indexing, which would add device synchronizations during edge creation.
    targets = torch.cat(
        [
            torch.where(left, blocks[:, :, :-1], blocks[:, :, 1:]).flatten(),
            torch.where(above, blocks[:, :-1, :], blocks[:, 1:, :]).flatten(),
            torch.where(above_left, blocks[:, :-1, :-1], blocks[:, 1:, 1:]).flatten(),
            torch.where(above_right, blocks[:, :-1, 1:], blocks[:, 1:, :-1]).flatten(),
        ]
    )
    parents = blocks.flatten()
    while True:
        source_roots, target_roots = parents[sources], parents[targets]
        merged = parents.clone()
        merged.scatter_reduce_(
            0, torch.maximum(source_roots, target_roots), torch.minimum(source_roots, target_roots), reduce="amin"
        )
        # Only roots are hooked and every parent decreases, so cycles cannot
        # form. Fully compress before hooking again: hooking non-roots could
        # detach an already-merged subtree.
        while True:
            compressed = merged[merged]
            if torch.equal(compressed, merged):
                break
            merged = compressed
        if torch.equal(merged, parents):
            break
        parents = merged

    labels = (parents.reshape(blocks.shape) + 1).repeat_interleave(2, -2).repeat_interleave(2, -1)
    return (labels[:, :height, :width] * mask).reshape(image.shape)
