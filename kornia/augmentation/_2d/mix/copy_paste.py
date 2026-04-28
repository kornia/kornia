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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation.utils import _validate_input_dtype
from kornia.constants import DataKey
from kornia.core.check import KORNIA_CHECK

__all__ = ["RandomCopyPaste"]


class RandomCopyPaste(MixAugmentationBaseV2):
    """Copy-paste augmentation — paste random instances from another image.

    For V1: takes a paired (image, mask) input and pastes mask>0 regions from
    a randomly-sampled OTHER batch element onto each target image.

    The source region is randomly scaled within ``scale_range`` and placed at a
    random position inside the target image before compositing.

    Args:
        scale_range: (min, max) random scale applied to the pasted region. Default (0.5, 1.5).
        p: per-sample probability of pasting. Default 0.5.
        same_on_batch: bool. Default False.

    Reference:
        Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation Method for
        Instance Segmentation" (CVPR 2021).

    Example:
        >>> import torch
        >>> aug = RandomCopyPaste(p=1.0)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> m = (torch.rand(4, 1, 64, 64) > 0.7).float()
        >>> y = aug(x, masks=m)
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        p: float = 0.5,
        same_on_batch: bool = False,
    ) -> None:
        super().__init__(p=p, p_batch=1.0, same_on_batch=same_on_batch)
        KORNIA_CHECK(
            len(scale_range) == 2 and scale_range[0] > 0 and scale_range[1] >= scale_range[0],
            f"scale_range must be (min, max) with 0 < min <= max, got {scale_range}.",
        )
        self.scale_range = scale_range
        # Override data_keys to IMAGE + MASK
        self.data_keys = [DataKey.INPUT, DataKey.MASK]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_scale(self, batch_size: int) -> torch.Tensor:
        """Return (B,) scales uniformly in scale_range."""
        lo, hi = self.scale_range
        return torch.empty(batch_size, device=self.device, dtype=torch.float32).uniform_(lo, hi)

    def _paste_one(
        self,
        target_img: torch.Tensor,
        source_img: torch.Tensor,
        source_mask: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paste source_mask>0 region from source_img into target_img.

        Args:
            target_img: (C, H, W) float tensor.
            source_img: (C, H, W) float tensor.
            source_mask: (1, H, W) float tensor — binary indicator.
            scale: scalar scale to resize source before pasting.

        Returns:
            Tuple of composited (C, H, W) image and updated (1, H, W) mask.
        """
        _, H, W = target_img.shape

        # Scale source image and mask
        new_h = max(1, round(H * scale))
        new_w = max(1, round(W * scale))

        src_img_scaled = F.interpolate(
            source_img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        src_mask_scaled = F.interpolate(source_mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)

        # Random top-left corner so the pasted patch overlaps the canvas
        max_top = H - 1
        max_left = W - 1
        top = int(torch.randint(0, max(1, max_top + 1), (1,)).item())
        left = int(torch.randint(0, max(1, max_left + 1), (1,)).item())

        # Crop the scaled source to the valid region
        src_h = min(new_h, H - top)
        src_w = min(new_w, W - left)

        if src_h <= 0 or src_w <= 0:
            return target_img, source_mask

        patch_img = src_img_scaled[:, :src_h, :src_w]
        patch_mask = src_mask_scaled[:, :src_h, :src_w]

        # Composite: where mask > 0, paste source
        binary = (patch_mask > 0.5).float()

        out_img = target_img.clone()
        out_img[:, top : top + src_h, left : left + src_w] = (
            binary * patch_img + (1.0 - binary) * target_img[:, top : top + src_h, left : left + src_w]
        )

        # Union mask: propagate pasted mask into target mask
        out_mask = source_mask.clone()  # start from target mask (passed in below)
        out_mask[:, top : top + src_h, left : left + src_w] = torch.max(
            source_mask[:, top : top + src_h, left : left + src_w], binary
        )

        return out_img, out_mask

    # ------------------------------------------------------------------
    # Public forward — overrides base entirely (like RandomTransplantation)
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        *input: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply copy-paste augmentation.

        Args:
            *input: image tensor(s) of shape (B, C, H, W).
            masks: optional mask tensor of shape (B, 1, H, W) or (B, H, W).
                   Regions where mask > 0 define which pixels to paste.
            params: pre-computed parameter dict (for replay).
            data_keys: unused — kept for API compatibility.

        Returns:
            If ``masks`` is None: augmented image tensor (B, C, H, W).
            If ``masks`` is given: list [augmented_image (B, C, H, W), augmented_mask (B, ?, H, W)].
        """
        if len(input) == 0:
            raise ValueError("At least one image tensor is required.")

        images = input[0]
        _validate_input_dtype(images, accepted_dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64])

        orig_dtype = images.dtype
        # Work in float32 for interpolation
        images_f = images.float()

        B, _, H, W = images_f.shape

        # Normalise mask shape to (B, 1, H, W) float32
        has_mask = masks is not None
        if has_mask:
            m = masks.float()
            if m.ndim == 3:
                m = m.unsqueeze(1)
            KORNIA_CHECK(m.ndim == 4, f"masks must be 3- or 4-D, got {m.ndim}-D.")
        else:
            # No mask — use full-image uniform mask (paste everything)
            m = torch.ones(B, 1, H, W, dtype=torch.float32, device=images_f.device)

        # Generate or reuse parameters
        if params is None:
            self._params = self.forward_parameters(images_f.shape)
        else:
            self._params = params

        batch_prob = self._params["batch_prob"]  # (B,)

        # Random permutation for source indices — each sample draws from another
        if "paste_pairs" not in self._params:
            # Build a permutation where i -> j with j != i where possible
            perm = torch.randperm(B, device=images_f.device)
            # Fix any position where perm[i] == i by swapping with next
            for i in range(B):
                if perm[i] == i:
                    swap = (i + 1) % B
                    perm[i], perm[swap] = perm[swap].clone(), perm[i].clone()
            self._params["paste_pairs"] = perm

        if "scales" not in self._params:
            self._params["scales"] = self._random_scale(B)

        paste_pairs = self._params["paste_pairs"].to(images_f.device)
        scales = self._params["scales"].to(images_f.device)

        out_images = images_f.clone()
        out_masks = m.clone()

        to_apply = batch_prob > 0.5

        for i in range(B):
            if not to_apply[i]:
                continue
            j = int(paste_pairs[i].item())
            scale = float(scales[i].item())

            composited_img, composited_mask = self._paste_one(
                target_img=images_f[i],
                source_img=images_f[j],
                source_mask=m[j],
                scale=scale,
            )
            # Merge composited mask with existing target mask
            out_images[i] = composited_img
            out_masks[i] = torch.max(out_masks[i], composited_mask)

        # Cast back to original dtype
        out_images = out_images.to(orig_dtype)
        out_masks = out_masks.to(masks.dtype if has_mask else orig_dtype)

        if has_mask:
            # Restore mask to original shape if it was 3-D on input
            if masks.ndim == 3:  # type: ignore[union-attr]
                out_masks = out_masks.squeeze(1)
            return [out_images, out_masks]

        return out_images

    # Satisfy abstract method requirements (not used via direct forward)
    def apply_transform(  # type: ignore[override]
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        maybe_flags: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return input
