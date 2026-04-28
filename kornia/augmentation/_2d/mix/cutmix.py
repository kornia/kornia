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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.constants import DataKey, DType
from kornia.geometry.bbox import bbox_to_mask, infer_bbox_shape
from kornia.geometry.boxes import Boxes


class RandomCutMixV2(MixAugmentationBaseV2):
    r"""Apply CutMix augmentation to a batch of torch.Tensor images.

    .. image:: _static/img/RandomCutMixV2.png

    Implementation for `CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features` :cite:`yun2019cutmix`.

    The function returns (inputs, labels), in which the inputs is the torch.Tensor that contains the mixup images
    while the labels is a :math:`(\text{num_mixes}, B, 3)` torch.Tensor that contains (label_permuted_batch, lambda)
    for each cutmix.

    The implementation referred to the following repository: `https://github.com/clovaai/CutMix-PyTorch
    <https://github.com/clovaai/CutMix-PyTorch>`_.

    Args:
        height: the width of the input image.
        width: the width of the input image.
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
        Tuple[torch.Tensor, torch.Tensor]:
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
        min_area: float = 1.0,
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
        self.min_area = min_area

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

    def apply_transform_boxes(self, input: Boxes, params: Dict[str, torch.Tensor], flags: Dict[str, Any]) -> Boxes:
        """Apply CutMix box remapping.

        For each mix, target image boxes that have sufficient visible area after the cut are kept (area
        remaining outside the cut rectangle must be >= min_area); source image boxes that intersect the
        cut rectangle are clipped to the cut bounds and added to the target.
        """
        # input._data: (B, N, 4, 2) in vertices_plus format
        # Convert to xyxy (xmin, ymin, xmax, ymax) for arithmetic; vertices_plus → xyxy_plus → subtract 1 offset
        boxes_v = input.to_tensor("xyxy_plus")  # (B, N, 4) xmin ymin xmax ymax (inclusive)
        device = boxes_v.device
        dtype = boxes_v.dtype

        out_boxes_v = boxes_v.clone()  # will be updated per mix

        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            # crop: (B, 4, 2), vertices: TL TR BR BL in (x,y)
            # Cut rect for each image: x1=crop[:,0,0], y1=crop[:,0,1], x2=crop[:,2,0], y2=crop[:,2,1]
            cx1 = crop[:, 0, 0].to(device=device, dtype=dtype)  # (B,)
            cy1 = crop[:, 0, 1].to(device=device, dtype=dtype)  # (B,)
            cx2 = crop[:, 2, 0].to(device=device, dtype=dtype)  # (B,)
            cy2 = crop[:, 2, 1].to(device=device, dtype=dtype)  # (B,)

            # out_boxes_v: (B, N, 4) — target boxes for this batch
            B, N, _ = out_boxes_v.shape

            # Expand cut coords to (B, N) for vectorised ops
            _cx1 = cx1.unsqueeze(1).expand(B, N)  # (B, N)
            _cy1 = cy1.unsqueeze(1).expand(B, N)
            _cx2 = cx2.unsqueeze(1).expand(B, N)
            _cy2 = cy2.unsqueeze(1).expand(B, N)

            # Target box coords (B, N)
            bx1 = out_boxes_v[..., 0]
            by1 = out_boxes_v[..., 1]
            bx2 = out_boxes_v[..., 2]
            by2 = out_boxes_v[..., 3]

            # Intersection of each target box with the cut rectangle
            ix1 = torch.max(bx1, _cx1)
            iy1 = torch.max(by1, _cy1)
            ix2 = torch.min(bx2, _cx2)
            iy2 = torch.min(by2, _cy2)

            inter_w = (ix2 - ix1).clamp(min=0.0)
            inter_h = (iy2 - iy1).clamp(min=0.0)
            inter_area = inter_w * inter_h  # (B, N)

            orig_w = (bx2 - bx1).clamp(min=0.0)
            orig_h = (by2 - by1).clamp(min=0.0)
            orig_area = orig_w * orig_h  # (B, N)

            # Visible area of target box after cut operation
            visible_area = orig_area - inter_area  # (B, N)

            # Zero out target boxes whose visible area falls below threshold
            drop_mask = visible_area < self.min_area  # (B, N)
            new_target = out_boxes_v.clone()
            new_target[drop_mask] = 0.0

            # ---- Source boxes: gather from permuted batch ----
            # pair: (B,) — pair[i] is the source image index for image i
            src_boxes_v = boxes_v.index_select(0, pair.to(device))  # (B, N, 4)

            sbx1 = src_boxes_v[..., 0]
            sby1 = src_boxes_v[..., 1]
            sbx2 = src_boxes_v[..., 2]
            sby2 = src_boxes_v[..., 3]

            # Intersect source boxes with the cut rectangle
            six1 = torch.max(sbx1, _cx1)
            siy1 = torch.max(sby1, _cy1)
            six2 = torch.min(sbx2, _cx2)
            siy2 = torch.min(sby2, _cy2)

            src_inter_w = (six2 - six1).clamp(min=0.0)
            src_inter_h = (siy2 - siy1).clamp(min=0.0)
            src_inter_area = src_inter_w * src_inter_h  # (B, N)

            # Keep source boxes that have positive intersection with the cut rect
            keep_src = src_inter_area >= self.min_area  # (B, N)

            # Clip source boxes to cut bounds
            clipped_src = torch.stack([six1, siy1, six2, siy2], dim=-1)  # (B, N, 4)
            clipped_src[~keep_src] = 0.0

            # Concatenate kept target boxes and clipped source boxes
            combined = torch.cat([new_target, clipped_src], dim=1)  # (B, 2*N, 4)

            out_boxes_v = combined

        # Convert back to Boxes (xyxy_plus → from_tensor)
        result = Boxes.from_tensor(out_boxes_v, mode="xyxy_plus", validate_boxes=False)
        return result

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], maybe_flags: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        height, width = input.size(2), input.size(3)

        # Use ``torch.where`` instead of boolean-indexed assignment.  The
        # original ``out[mask] = src[mask]`` does two scatter/gather kernels
        # per mix and dominates the cost (≈ 95% of forward time on a
        # 8×3×512×512 tensor).  ``torch.where`` is a single elementwise kernel
        # that broadcasts the (B, 1, H, W) mask across channels.
        out_inputs = input
        for pair, crop in zip(params["mix_pairs"], params["crop_src"]):
            input_permute = input.index_select(dim=0, index=pair.to(input.device))
            # (B, 1, H, W) bool mask broadcasts over the channel axis in ``where``.
            mask = bbox_to_mask(crop, width, height).bool().unsqueeze(dim=1)
            out_inputs = torch.where(mask, input_permute, out_inputs)

        return out_inputs
