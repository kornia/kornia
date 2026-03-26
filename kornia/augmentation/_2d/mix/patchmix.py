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


class PatchMix(MixAugmentationBaseV2):
    r"""PatchMix augmentation.

    Replaces a random patch in each image of a batch with the corresponding
    region from a randomly chosen different image in the batch.

    Args:
        alpha: Hyperparameter for the Beta distribution used to generate the
            mixing parameter lambda.
        patch_size: The size of the square patch to mix.
        p: Probability for applying the augmentation.
        same_on_batch: Apply the same transformation across the batch.
        keepdim: Whether to keep the output shape the same as input ``True``
            or broadcast it to the batch form ``False``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_size: int = 16,
        p: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.alpha = alpha
        self.patch_size = patch_size

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_shape[0],))
        try:
            lam = lam.to(device=self.device, dtype=self.dtype)
        except (AttributeError, RuntimeError):
            pass
        return {"lam": lam}

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], extra_args: Dict[str, Any]
    ) -> torch.Tensor:
        B, _, H, W = input.shape
        lam = params["lam"].to(input.device)
        idx = torch.randperm(B, device=input.device)
        out = input.clone()
        for i in range(B):
            # Random patch coordinates
            y = int(torch.randint(0, H - self.patch_size + 1, (), device=input.device).item())
            x = int(torch.randint(0, W - self.patch_size + 1, (), device=input.device).item())
            out[i, :, y : y + self.patch_size, x : x + self.patch_size] = input[
                idx[i], :, y : y + self.patch_size, x : x + self.patch_size
            ]
        
        # Expose mix parameters via params as expected by MixAugmentationBaseV2
        params["idx"] = idx
        params["lam"] = lam
        return out
