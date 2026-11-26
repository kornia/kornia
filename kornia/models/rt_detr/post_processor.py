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

"""Post-processor for the RT-DETR model."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from kornia.contrib.object_detection import BoxFiltering


def mod(a: torch.Tensor, b: int) -> torch.Tensor:
    """Compute the element-wise remainder of tensor `a` divided by integer `b`.

    This function requires `a` to be a `torch.Tensor` and `b` to be an `int`.
    It returns a `torch.Tensor` with the same shape/device as `a`. The
    implementation uses `a % b` (equivalent to `torch.remainder(a, b)`).

    Args:
        a (torch.Tensor): Dividend torch.tensor(any numeric dtype).
        b (int): Divisor (must be non-zero).

    Returns:
        torch.Tensor: Element-wise remainder of `a` divided by `b`.

    Examples:
        >>> mod(torch.tensor(7), 3)
        tensor(1)
        >>> mod(torch.tensor([7, -1, 2]), 3)
        tensor([1, 2, 2])
    """
    return a % b


# TODO: deprecate the confidence threshold and add the num_top_queries as a parameter and num_classes as a parameter
class DETRPostProcessor(nn.Module):
    """Convert raw DETR model outputs into final bounding box detections.

    This module applies the sigmoid function to the class logits, as RT-DETR was trained with a focal loss, keeps the
    ``num_top_queries`` highest (query, class) scores of each image, and transforms the normalized
    ``(cx, cy, w, h)`` boxes into ``(x, y, w, h)`` in the pixel coordinate system of each input image, scaled by that
    image's own size.

    Args:
        num_classes: The number of object classes.
        confidence_threshold: The threshold to filter out low-confidence detections.
        num_top_queries: The number of top queries to consider for each image.
        confidence_filtering: Whether to apply confidence-based filtering.
        filter_as_zero: If True, boxes below the confidence threshold are set to zero instead of being removed.
    """

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        num_classes: int = 80,
        num_top_queries: int = 300,
        confidence_filtering: bool = True,
        filter_as_zero: bool = False,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.confidence_filtering = confidence_filtering
        self.num_top_queries = num_top_queries
        self.box_filtering = BoxFiltering(
            torch.tensor(confidence_threshold) if confidence_threshold is not None else None,
            filter_as_zero=filter_as_zero,
        )

    def forward(
        self, logits: torch.Tensor, boxes: torch.Tensor, original_sizes: torch.Tensor
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Post-process outputs from DETR.

        Args:
            logits: tensor with shape ``(N, Q, K)``, where :math:`N` is the batch size, :math:`Q` is the number of
                queries, :math:`K` is the number of classes.
            boxes: tensor with shape ``(N, Q, 4)``, where :math:`N` is the batch size, :math:`Q` is the number of
                queries.
            original_sizes: tensor with shape ``(N, 2)``, where :math:`N` is the batch size and each element
                represents the image size of (img_height, img_width).

        Returns:
            Processed detections, each a row (class_id, confidence_score, x, y, w, h). A tensor of shape
            ``(N, T, 6)``, :math:`T` = ``num_top_queries``, when ``confidence_filtering`` is ``False`` or
            ``confidence_threshold`` is 0; with ``filter_as_zero``, the same tensor with the detections at or below the
            threshold set to zero. Otherwise a list of :math:`N` tensors of shape ``(D, 6)``, where :math:`D` is the
            number of detections above the threshold in that image.

        """
        # NOTE: consider using kornia BoundingBox
        # NOTE: consider having a separate function to convert the detections to a list of bounding boxes

        # https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/post_process.py#L446
        # box format is cxcywh
        # convert to xywh
        # bboxes[..., :2] -= bboxes[..., 2:] * 0.5  # in-place operation is not torch.compile()-friendly
        # TODO: replace using kornia BoundingBox
        cxcy, wh = boxes[..., :2], boxes[..., 2:]
        boxes_xy = torch.cat([cxcy - wh * 0.5, wh], -1)

        # (B, 2) (H, W) -> (B, 1, 4) (W, H, W, H): each image is scaled by its own size
        sizes_wh = original_sizes.flip(-1).repeat(1, 2).unsqueeze(1).to(boxes_xy.dtype)

        boxes_xy = boxes_xy * sizes_wh
        scores = logits.sigmoid()  # RT-DETR was trained with focal loss. thus sigmoid is used instead of softmax

        # retrieve the boxes with the highest score for each class
        # https://github.com/lyuwenyu/RT-DETR/blob/b6bf0200b249a6e35b44e0308b6058f55b99696b/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py#L55-L62
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
        labels = mod(index, self.num_classes)
        index = index // self.num_classes
        boxes = boxes_xy.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes_xy.shape[-1]))

        all_boxes = torch.cat([labels[..., None], scores[..., None], boxes], -1)

        if not self.confidence_filtering or self.confidence_threshold == 0:
            return all_boxes

        return self.box_filtering(all_boxes, self.confidence_threshold)
