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

from typing import Any, Dict, List, Optional, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey


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
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self.alpha = alpha
        self.patch_size = patch_size
        self._param_generator = rg.PatchMixGenerator(alpha, patch_size, p)

    def apply_transform_class(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        B, C, H, W = params["batch_shape"]
        idx = params["mix_pairs"].to(input.device)

        # Calculate area-based lambda for labels
        area_ratio = (float(self.patch_size) ** 2) / (float(H) * float(W))
        lam = torch.full((B,), 1.0 - area_ratio, device=input.device, dtype=input.dtype)

        # Prepare mixed labels: [label_orig, label_perm, lam]
        labels_permute = input.index_select(dim=0, index=idx)

        out = torch.stack(
            [
                input.to(dtype=input.dtype),
                labels_permute.to(dtype=input.dtype),
                lam.to(dtype=input.dtype),
            ],
            -1,
        )

        return out

    def apply_non_transform_class(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        B = input.shape[0]
        lam = torch.ones((B,), device=input.device, dtype=input.dtype)
        out = torch.stack(
            [
                input.to(dtype=input.dtype),
                input.to(dtype=input.dtype),
                lam.to(dtype=input.dtype),
            ],
            -1,
        )
        return out

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return self._param_generator(batch_shape, self.same_on_batch)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], extra_args: Dict[str, Any]
    ) -> torch.Tensor:
        B, C, H, W = input.shape
        idx = params["mix_pairs"].to(input.device)
        patch_coords = params["patch_coords"].to(input.device)

        out = input.clone()
        for i in range(B):
            x, y = patch_coords[i]
            x, y = int(x.item()), int(y.item())
            out[i, :, y : y + self.patch_size, x : x + self.patch_size] = input[
                idx[i], :, y : y + self.patch_size, x : x + self.patch_size
            ]

        return out
