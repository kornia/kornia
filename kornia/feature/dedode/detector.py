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
from torch import nn


class DeDoDeDetector(nn.Module):
    """Implement the DeDoDe detector for keypoint learning and localization.

    Args:
        encoder: The backbone encoder module.
        decoder: The decoder module for probability map generation.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Run this DeDoDe module forward.

        Args:
            images: Input image batch with shape ``(B, C, H, W)``.

        Returns:
            Keypoint detection logits with shape ``(B, 1, H, W)``, in at least float32: float32 for half-precision
            ``images``, float64 for float64 ones.
        """
        dtype = images.dtype
        features, sizes = self.encoder(images)
        context = None
        logits = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(feature_map, context=context, scale=scale)
            # The logits are accumulated and returned in at least float32: rounding them to a half dtype would reorder
            # the top-k that `DeDoDe.detect` takes from their softmax.
            delta_logits = delta_logits.to(torch.promote_types(delta_logits.dtype, torch.float32))
            logits = delta_logits if logits is None else logits + delta_logits
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False).to(dtype)
        return logits
