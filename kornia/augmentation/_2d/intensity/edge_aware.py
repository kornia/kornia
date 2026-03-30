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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.filters import canny, sobel


class EdgeAwareAugmentation(IntensityAugmentationBase2D):
    r"""Edge-aware augmentation for images.

    This module wraps another intensity augmentation (brightness, blur, etc.) and
    applies it more strongly in smooth image regions while reducing its effect
    near sharp edges or boundaries.

    Why use this?
    - Avoids over-augmenting edges, which helps preserve object structure.
    - Offers controlled, interpretable behavior via `edge_weight` and mode.
    - Works with existing Kornia augmentations as a drop-in wrapper.

    The augmentation works by:
    1. Computing an edge map from the input image using either Sobel or Canny edge detection
    2. Creating a modulation mask that reduces augmentation intensity near edges
    3. Blending the original image with the augmented version based on the modulation mask

    In "soft" mode, the modulation is continuous: modulation = 1 - edge_weight * edge_map
    In "hard" mode, the modulation is binary: edges are thresholded and scaled by edge_weight

    .. image:: _static/img/EdgeAwareAugmentation.png

    Args:
        base_aug: The base augmentation module to apply (e.g., RandomBrightness, RandomGaussianBlur).
        edge_detector: Edge detection method, either "sobel" or "canny". Sobel is faster but Canny may be more accurate.
        mode: Modulation mode, either "soft" (continuous weighting) or "hard" (binary masking).
            - "soft": Applies continuous modulation where augmentation strength decreases linearly with edge strength
            - "hard": Applies binary modulation where regions above edge threshold get reduced augmentation
        edge_weight: Weight for edge modulation, must be in [0.0, 1.0]. Higher values preserve edges more strongly.
            At 0.0, no edge preservation occurs (normal augmentation). At 1.0, edges are fully preserved.
        detach_edges: Whether to detach edge computation from the gradient graph. Default True for efficiency.
        same_on_batch: Apply the same transformation across the batch.
        p: Probability of applying the transformation.
        keepdim: Whether to keep the output shape the same as input (True) or broadcast to batch form (False).

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> from kornia.augmentation import RandomGaussianBlur, EdgeAwareAugmentation
        >>> base_aug = RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        >>> aug = EdgeAwareAugmentation(base_aug, edge_weight=0.3, p=1.0)
        >>> img = torch.rand(1, 3, 32, 32)
        >>> out = aug(img)
        >>> out.shape
        torch.Size([1, 3, 32, 32])

    """

    def __init__(
        self,
        base_aug: nn.Module,
        edge_detector: str = "sobel",
        mode: str = "soft",
        edge_weight: float = 0.3,
        detach_edges: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.base_aug = base_aug
        self.edge_detector = edge_detector
        self.mode = mode
        self.edge_weight = edge_weight
        self.detach_edges = detach_edges

        # Validate edge_detector
        if edge_detector not in ["sobel", "canny"]:
            raise ValueError(f"edge_detector must be 'sobel' or 'canny', got '{edge_detector}'")

        # Validate mode
        if mode not in ["soft", "hard"]:
            raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")

        # Validate edge_weight range
        if not (0.0 <= edge_weight <= 1.0):
            raise ValueError(f"edge_weight must be in [0.0, 1.0], got {edge_weight}")

    def _compute_edge_map(self, input: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude map from input image.

        Args:
            input: Input image tensor of shape (B, C, H, W).

        Returns:
            Edge magnitude map of shape (B, 1, H, W) with values in [0, 1].
        """
        if self.edge_detector == "sobel":
            # Sobel returns (B, C, H, W) - compute magnitude per channel then average
            edge_map = sobel(input, normalized=True, eps=1e-6)
            # Average across channels to get single-channel edge map
            edge_map = edge_map.mean(dim=1, keepdim=True)
        else:  # canny
            # Canny returns tuple (edges, magnitude)
            _, edge_map = canny(input, low_threshold=0.1, high_threshold=0.2)
            edge_map = edge_map.mean(dim=1, keepdim=True)

        # Normalize to [0, 1] range
        edge_map = edge_map - edge_map.min()
        max_val = edge_map.max()
        if max_val > 0:
            edge_map = edge_map / max_val

        if self.detach_edges:
            edge_map = edge_map.detach()

        return edge_map

    def _compute_modulation_mask(self, edge_map: torch.Tensor) -> torch.Tensor:
        """Compute modulation mask from edge map.

        Args:
            edge_map: Edge magnitude map of shape (B, 1, H, W).

        Returns:
            Modulation mask of shape (B, 1, H, W) with values in [0, 1].
            Lower values near edges (high edge_map) mean less augmentation.
        """
        if self.mode == "soft":
            # Soft mode: continuous weighting based on edge strength
            # modulation = 1 - edge_weight * edge_map
            # At edges (edge_map=1): modulation = 1 - edge_weight
            # In smooth regions (edge_map=0): modulation = 1
            modulation = 1.0 - self.edge_weight * edge_map
        else:  # hard
            # Hard mode: binary masking
            # Threshold edge_map at 0.5, apply edge_weight as scaling factor
            threshold = 0.5
            binary_edges = (edge_map > threshold).float()
            # In edge regions: modulation = 1 - edge_weight
            # In non-edge regions: modulation = 1
            modulation = 1.0 - self.edge_weight * binary_edges

        return modulation

    def apply_transform(
        self,
        input: torch.Tensor,
        params: Dict[str, torch.Tensor],
        flags: Dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute edge map
        edge_map = self._compute_edge_map(input)

        # Compute modulation mask
        modulation = self._compute_modulation_mask(edge_map)

        # Apply base augmentation
        augmented = self.base_aug(input)

        # Blend original and augmented based on modulation mask
        # output = modulation * original + (1 - modulation) * augmented
        # At edges: more original preserved
        # In smooth regions: more augmentation applied
        output = modulation * input + (1.0 - modulation) * augmented

        return output
