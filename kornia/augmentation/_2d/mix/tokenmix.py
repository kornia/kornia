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

from typing import Any, Dict, Optional

import torch

from .base import MixAugmentationBaseV2


class TokenMix(MixAugmentationBaseV2):
    r"""Apply token-based mixing augmentation to a batch of images.

    Inspired by token-mixing strategies in vision transformers, this augmentation
    replaces a subset of non-overlapping image patches ("tokens") from each image
    with the corresponding patches from another randomly selected image in the batch.
    The number of tokens replaced is controlled by a Beta-distributed mixing coefficient
    ``lam``, so that ``alpha`` has a meaningful effect on the strength of mixing.

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
    ) -> None:
        """Initialise TokenMix augmentation.

        Args:
            alpha: Beta distribution concentration parameter. Default: ``1.0``.
            num_tokens: Number of token rows/columns per spatial axis. Default: ``8``.
            p: Batch-level application probability. Default: ``1.0``.
            same_on_batch: Apply identical transform to every item in the batch. Default: ``False``.
            keepdim: Keep spatial shape of the output. Default: ``False``.

        """
        super().__init__(p=p, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.alpha = alpha
        self.num_tokens = num_tokens

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        """Generate per-batch mixing parameters.

        Samples a Beta-distributed ``lam`` per image, converts it to a token
        count, and draws a random batch permutation. All values are stored in
        the returned ``params`` dict so that ``apply_transform`` can use them
        without re-sampling.

        Args:
            batch_shape: Shape of the input batch, i.e. ``(B, C, H, W)``.

        Returns:
            Dict with keys:
                - ``"lam"``: mixing coefficients, shape ``(B,)``
                - ``"num_mix_tokens"``: number of tokens to replace per image, shape ``(B,)``
                - ``"batch_perm"``: random permutation indices, shape ``(B,)``

        """
        B = batch_shape[0]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,))
        try:
            lam = lam.to(self.device).to(self.dtype)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass

        # Map lam in [0, 1] to a per-sample token count in [1, num_tokens^2].
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
        """Replace token-grid patches with patches from a permuted image.

        Divides each image into a ``num_tokens x num_tokens`` non-overlapping
        grid of patches and replaces the first ``num_mix_tokens[i]`` patches
        (in raster order) of image *i* with the corresponding patches from the
        permuted image. The batch dimension is processed with vectorised
        indexing; only the token loop is a Python-level iteration.

        Args:
            input: Batch of images, shape ``(B, C, H, W)``.
            params: Dict produced by :meth:`generate_parameters`. Must contain
                ``"num_mix_tokens"`` and ``"batch_perm"``.
            maybe_flags: Unused augmentation flags (kept for API compatibility).

        Returns:
            torch.Tensor: Augmented batch, shape ``(B, C, H, W)``.

        Raises:
            ValueError: If ``H`` or ``W`` are not divisible by ``num_tokens``,
                or if ``num_tokens`` is larger than ``min(H, W)``.

        """
        B, _C, H, W = input.shape
        token_h = H // self.num_tokens
        token_w = W // self.num_tokens

        if token_h == 0 or token_w == 0:
            raise ValueError(
                f"TokenMix: num_tokens={self.num_tokens} exceeds spatial dimensions "
                f"(H={H}, W={W}). num_tokens must be <= min(H, W)."
            )
        if H % self.num_tokens != 0 or W % self.num_tokens != 0:
            raise ValueError(
                f"TokenMix: input size (H={H}, W={W}) must be divisible by "
                f"num_tokens={self.num_tokens}."
            )

        idx = params["batch_perm"].to(input.device)
        num_mix_tokens = params["num_mix_tokens"].to(input.device)  # (B,)

        # Pre-fetch the permuted images once — no outer Python loop over batch.
        shuffled = input[idx]  # (B, C, H, W)
        out = input.clone()

        # Replace tokens in raster order up to the per-sample count.
        # The inner loop is over at most num_tokens^2 iterations (typically small).
        token_idx = 0
        for ty in range(self.num_tokens):
            y = ty * token_h
            for tx in range(self.num_tokens):
                x = tx * token_w
                # Boolean mask: which images in the batch should get this token replaced.
                mask = (num_mix_tokens > token_idx).view(B, 1, 1, 1)
                out[:, :, y : y + token_h, x : x + token_w] = torch.where(
                    mask,
                    shuffled[:, :, y : y + token_h, x : x + token_w],
                    out[:, :, y : y + token_h, x : x + token_w],
                )
                token_idx += 1

        return out
