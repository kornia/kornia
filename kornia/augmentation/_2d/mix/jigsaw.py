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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey

__all__ = ["RandomJigsaw"]


class RandomJigsaw(MixAugmentationBaseV2):
    r"""RandomJigsaw augmentation.

    .. image:: _static/img/RandomJigsaw.png

    Make Jigsaw puzzles for each image individually. To mix with different images in a
    batch, referring to :class:`kornia.augmentation.RandomMosaic`.

    See the Convention block on :class:`~kornia.augmentation.MixAugmentationBaseV2`.

    Args:
        grid: the Jigsaw puzzle grid. e.g. (2, 2) means
            each output will mix image patches in a 2x2 grid.
        ensure_perm: reject the image-preserving permutation ``arange(N).view(rows, columns).T.flatten()`` when
            drawing, so a selected sample is never returned unchanged. For a single-row or single-column grid that
            permutation is ``[0, ..., N - 1]``; see the Convention block. A ``1 x 1`` grid has no other
            permutation, so it raises ``ValueError`` unless ``ensure_perm=False``.
        data_keys: the input type sequential for applying augmentations. Only "input" and "image" are
            implemented; see the Convention block.
        p: probability of applying the transformation to each sample.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
            to the batch form ``False``.

    Convention:
        - ``grid=(rows, columns)`` partitions each image independently. Both image dimensions must be exactly
          divisible by their corresponding grid entries; any other input raises ``RuntimeError``, whether or not
          the gate selects a sample. The output preserves the input shape. An entry's position selects the
          destination cell in column-major order -- for a ``2 x 2`` grid, entries ``[0, 2]`` become the top row
          and entries ``[1, 3]`` the bottom row -- while the entry's value indexes the source patch in row-major
          order. The two orders differ unless the grid has a single row or column; where they differ, the
          identity permutation does not reproduce the image -- on a square grid it transposes the patch grid. The
          image-preserving permutation is ``arange(N).view(rows, columns).T.flatten()``: ``[0, 2, 1, 3]`` for a
          ``2 x 2`` grid.
        - ``p`` is a per-sample gate. With ``same_on_batch=True`` the batch shares one gate draw and one patch
          permutation; with it false, each sample receives an independent gate and permutation. This class
          implements image mixing only; requesting another data key raises ``NotImplementedError`` whatever the
          gate, as described on the base.

    Examples:
        >>> jigsaw = RandomJigsaw((4, 4))
        >>> input = torch.randn(8, 3, 256, 256)
        >>> out = jigsaw(input)
        >>> out.shape
        torch.Size([8, 3, 256, 256])

    """

    def __init__(
        self,
        grid: Tuple[int, int] = (4, 4),
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
        ensure_perm: bool = True,
    ) -> None:
        super().__init__(p=p, p_batch=1.0, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self._param_generator = rg.JigsawGenerator(grid, ensure_perm)
        self.flags = {"grid": grid}

    def transform_tensor(
        self, input: torch.Tensor, *, shape: Optional[torch.Tensor] = None, match_channel: bool = True
    ) -> torch.Tensor:
        input = super().transform_tensor(input, shape=shape, match_channel=match_channel)
        h, w = input.shape[-2:]
        if h % self.flags["grid"][0] or w % self.flags["grid"][1]:
            raise RuntimeError(f"Input height and width {(h, w)} must be divisible by grid {self.flags['grid']}.")
        return input

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        # Operate on the full batch: params are full-batch sized and the base class
        # where-blends this result with the non-transformed branch per `batch_prob`.
        b, c, h, w = input.shape
        perm = params["permutation"]
        piece_size_h, piece_size_w = input.shape[-2] // self.flags["grid"][0], input.shape[-1] // self.flags["grid"][1]
        # Convert to C BxN H' W'
        input = (
            input.unfold(2, piece_size_h, piece_size_h)
            .unfold(3, piece_size_w, piece_size_w)
            .reshape(b, c, -1, piece_size_h, piece_size_w)
            .permute(1, 0, 2, 3, 4)
            .reshape(c, -1, piece_size_h, piece_size_w)
        )
        perm = (perm + torch.arange(0, b, device=perm.device)[:, None] * perm.shape[1]).view(-1)
        input = input[:, perm, :, :]
        input = (
            input.reshape(-1, b, self.flags["grid"][1], h, piece_size_w)
            .permute(0, 1, 2, 4, 3)
            .reshape(-1, b, w, h)
            .permute(0, 1, 3, 2)
            .permute(1, 0, 2, 3)
        )
        return input
