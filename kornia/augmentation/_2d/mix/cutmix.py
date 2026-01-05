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
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey, DType
from kornia.geometry.bbox import bbox_to_mask, infer_bbox_shape


class RandomCutMixV2(MixAugmentationBaseV2):
    r"""Apply CutMix augmentation to a batch of torch.tensor images.

    .. image:: _static/img/RandomCutMixV2.png

    Implementation for `CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features` :cite:`yun2019cutmix`.

    The function returns (inputs, labels), in which the inputs is the torch.tensor that contains the mixup images
    while the labels is a :math:`(\text{num_mixes}, B, 3)` torch.tensor that contains (label_permuted_batch, lambda)
    for each cutmix.

    The implementation referred to the following repository: `https://github.com/clovaai/CutMix-PyTorch
    <https://github.com/clovaai/CutMix-PyTorch>`_.

    Args:
        p: probability for applying an augmentation to a batch. This param controls the augmentation
            probabilities batch-wisely.
        num_mix: cut mix times.
        beta: hyperparameter for generating cut size from beta distribution.
            Beta cannot be set to 0 after torch 1.8.0. If None, it will be set to 1.
        cut_size: controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch: apply the same transformation across the batch.
            This flag will not maintain permutation order.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        use_correct_lambda: if True, compute lambda according to the CutMix paper
            (`lam = 1 - area_ratio`). Defaults to False (`lam = area_ratio`) for backward compatibility,
            but will raise a deprecation warning when False.

    Inputs:
        - Input image tensors, shape of :math:`(B, C, H, W)`.
        - Raw labels, shape of :math:`(B)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - Adjusted image, shape of :math:`(B, C, H, W)`.
            - Raw labels, permuted labels and lambdas for each mix, shape of :math:`(B, num_mix, 3)`.

    Note:
        This implementation would randomly cutmix images in a batch. Ideally, the larger batch size would be preferred.

    Examples:
        >>> rng = torch.manual_seed(3)
        >>> input = torch.rand(2, 1, 3, 3)
        >>> input[0] = torch.ones((1, 3, 3))
        >>> label = torch.tensor([0, 1])
        >>> cutmix = RandomCutMixV2(data_keys=["input", "class"], use_correct_lambda=True)
        >>> cutmix(input, label)
        [tensor([[[[0.8879, 0.4510, 1.0000],
                  [0.1498, 0.4015, 1.0000],
                  [1.0000, 1.0000, 1.0000]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[1.0000, 1.0000, 0.7995],
                  [1.0000, 1.0000, 0.0542],
                  [0.4594, 0.1756, 0.9492]]]]), tensor([[[0.0000, 1.0000, 0.5556],
                 [1.0000, 0.0000, 0.5556]]])]

    """

    def __init__(
        self,
        num_mix: int = 1,
        cut_size: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[torch.Tensor, float]] = None,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
        use_correct_lambda: bool = False,
    ) -> None:
        super().__init__(p=1.0, p_batch=p, same_on_batch=same_on_batch, keepdim=keepdim, data_keys=data_keys)
        self._param_generator: rg.CutmixGenerator = rg.CutmixGenerator(cut_size, beta, num_mix, p=p)

        self.use_correct_lambda = use_correct_lambda
        if not self.use_correct_lambda:
            warnings.warn(
                "RandomCutMixV2 currently uses the old (inconsistent) lambda computation. "
                "Set `use_correct_lambda=True` to align with the original CutMix paper. "
                "This default will change in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

    def apply_transform_class(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Dict[str, Any]
    ) -> torch.Tensor:
        height, width = params["image_shape"]

        out_labels = []
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            labels_permute = input.index_select(dim=0, index=pair.to(input.device))
            w, h = infer_bbox_shape(crop)

            lam_val = w.to(input.dtype) * h.to(input.dtype) / (width * height)
            lam = 1 - lam_val if self.use_correct_lambda else lam_val

            out_labels.append(
                torch.stack(
                    [
                        input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                        labels_permute.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                        lam.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                    ],
                    1,
                )
            )

        return torch.stack(out_labels, 0)

    def apply_non_transform_class(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        out_labels = []
        lam = torch.zeros((len(input)), device=input.device, dtype=DType.to_torch(int(params["dtype"].item())))
        for _ in range(self._param_generator.num_mix):
            out_labels.append(
                torch.stack(
                    [
                        input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                        input.to(device=input.device, dtype=DType.to_torch(int(params["dtype"].item()))),
                        lam,
                    ],
                    1,
                )
            )

        return torch.stack(out_labels, 0)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        height, width = input.size(2), input.size(3)

        out_inputs = input.clone()
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            input_permute = input.index_select(dim=0, index=pair.to(input.device))
            # compute mask to match input shape
            mask = bbox_to_mask(crop, width, height).bool().unsqueeze(dim=1).repeat(1, input.size(1), 1, 1)
            out_inputs[mask] = input_permute[mask]

        return out_inputs
