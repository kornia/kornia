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

from typing import Any, Dict, List, Optional, Union

import torch

from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey


class TokenMix(MixAugmentationBaseV2):
    r"""Apply token-based mixing augmentation to a batch of images.

    .. image:: _static/img/TokenMix.png

    Inspired by token-mixing strategies in vision transformers, this augmentation
    replaces a subset of non-overlapping image patches ("tokens") from each image
    with the corresponding patches from another randomly selected image in the batch.
    The number of tokens replaced is controlled by a Beta-distributed mixing coefficient
    ``lam``, so that ``alpha`` has a meaningful effect on the strength of mixing.

    Implementation for `TokenMix: Rethinking Image Mixing for Data Augmentation
    in Vision Transformers` :cite:`ji2022tokenmix`.

    Args:
        alpha: concentration parameter for the Beta distribution used to sample
            the per-sample mixing coefficient ``lam``. Higher values produce ``lam``
            values closer to 0.5 (more mixing); lower values push ``lam`` toward 0 or 1.
        num_tokens: number of non-overlapping token rows (and columns) per spatial axis.
            The image is divided into a ``num_tokens x num_tokens`` grid. Both ``H`` and
            ``W`` must be divisible by ``num_tokens``.
        p: probability of applying the augmentation to the batch.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input.
        data_keys: the input types for applying augmentations.

    Convention:
        - ``p`` is a single batch-wide gate.
        - ``same_on_batch=True`` shares the mixing coefficient, token count, and replaced
          token positions across the batch.
        - Partner selection remains a random batch permutation; self-pairing is allowed,
          consistent with MixUp.
        - Both ``H`` and ``W`` must be divisible by ``num_tokens``, and ``num_tokens <= min(H, W)``.
        - Only image input is implemented.

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.

    Returns:
        torch.Tensor: Augmented images with shape :math:`(B, C, H, W)`.

    Examples:
        >>> aug = TokenMix(alpha=1.0, num_tokens=4)
        >>> x = torch.rand(2, 3, 32, 32)
        >>> out = aug(x)
        >>> out.shape
        torch.Size([2, 3, 32, 32])

    """

    def __init__(
        self,
        alpha: float = 1.0,
        num_tokens: int = 8,
        p: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> None:
        """Initialise TokenMix augmentation.

        Args:
            alpha: Beta distribution concentration parameter. Default: ``1.0``.
            num_tokens: Number of token rows/columns per spatial axis. Default: ``8``.
            p: Batch-level application probability. Default: ``1.0``.
            same_on_batch: Apply identical transform across the batch. Default: ``False``.
            keepdim: Keep spatial shape of the output. Default: ``False``.
            data_keys: List of target data types. Default: ``None``.

        """
        super().__init__(
            p=1.0,
            p_batch=p,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            data_keys=data_keys,
        )
        self.alpha = alpha
        self.num_tokens = num_tokens

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        """Generate per-batch mixing parameters.

        Samples a Beta-distributed ``lam`` per image (or single shared value when
        ``same_on_batch=True``), converts it to a token count, and draws a random
        batch permutation. Partner selection remains a random batch permutation,
        allowing self-pairing. All values are stored in the returned ``params`` dict
        so that ``apply_transform`` can use them without re-sampling.

        Args:
            batch_shape: Shape of the input batch, i.e. ``(B, C, H, W)``.

        Returns:
            Dict with keys:
                - ``"lam"``: mixing coefficients, shape ``(B,)``
                - ``"num_mix_tokens"``: number of tokens to replace per image, shape ``(B,)``
                - ``"batch_perm"``: random permutation indices, shape ``(B,)``

        """
        B = batch_shape[0]
        if B == 0:
            return {
                "lam": torch.empty((0,)),
                "num_mix_tokens": torch.empty((0,), dtype=torch.long),
                "batch_perm": torch.empty((0,), dtype=torch.long),
            }

        sample_size = 1 if self.same_on_batch and B > 0 else B
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((sample_size,))
        if self.same_on_batch:
            lam = lam.repeat(B)

        total_tokens = self.num_tokens * self.num_tokens
        num_mix_tokens = (lam * total_tokens).round().clamp(min=1.0, max=float(total_tokens)).long()

        batch_perm = torch.randperm(B)
        return {"lam": lam, "num_mix_tokens": num_mix_tokens, "batch_perm": batch_perm}

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        maybe_flags: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Replace token-grid patches with patches from a permuted image."""
        B, _C, H, W = input.shape
        token_h = H // self.num_tokens
        token_w = W // self.num_tokens

        if token_h == 0 or token_w == 0:
            raise ValueError(
                f"TokenMix: num_tokens={self.num_tokens} exceeds spatial dimensions "
                f"(H={H}, W={W}). num_tokens must be <= min(H, W)."
            )
        if H % self.num_tokens != 0 or W % self.num_tokens != 0:
            raise ValueError(f"TokenMix: input size (H={H}, W={W}) must be divisible by num_tokens={self.num_tokens}.")

        idx = params["batch_perm"].to(input.device)
        num_mix_tokens = params["num_mix_tokens"].to(input.device)

        shuffled = input[idx]
        out = input.clone()

        token_idx = 0
        for ty in range(self.num_tokens):
            y = ty * token_h
            for tx in range(self.num_tokens):
                x = tx * token_w
                mask = (num_mix_tokens > token_idx).view(B, 1, 1, 1)
                out[:, :, y : y + token_h, x : x + token_w] = torch.where(
                    mask,
                    shuffled[:, :, y : y + token_h, x : x + token_w],
                    out[:, :, y : y + token_h, x : x + token_w],
                )
                token_idx += 1

        return out
