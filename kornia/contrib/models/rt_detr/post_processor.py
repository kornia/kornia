"""Post-processor for the RT-DETR model."""

from __future__ import annotations

from typing import Optional, Union

import torch

from kornia.core import Module, Tensor, concatenate, tensor
from kornia.models.detection.utils import BoxFiltering


def mod(a: Tensor, b: int) -> Tensor:
    """Compute the modulo operation for two numbers.

    This function calculates the remainder of the division of 'a' by 'b'
    using the formula: a - (a // b) * b, which is equivalent to the modulo operation.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        The remainder of a divided by b.

    Example:
        >>> mod(7, 3)
        1
    """
    return a - (a // b) * b


# TODO: deprecate the confidence threshold and add the num_top_queries as a parameter and num_classes as a parameter
class DETRPostProcessor(Module):
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
            tensor(confidence_threshold) if confidence_threshold is not None else None, filter_as_zero=filter_as_zero
        )

    def forward(self, logits: Tensor, boxes: Tensor, original_sizes: Tensor) -> Union[Tensor, list[Tensor]]:
        """Post-process outputs from DETR.

        Args:
            logits: tensor with shape :math:`(N, Q, K)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries, :math:`K` is the number of classes.
            boxes: tensor with shape :math:`(N, Q, 4)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries.
            original_sizes: tensor with shape :math:`(N, 2)`, where :math:`N` is the batch size and each element
                represents the image size of (img_height, img_width).

        Returns:
            Processed detections. For each image, the detections have shape (D, 6), where D is the number of detections
            in that image, 6 represent (class_id, confidence_score, x, y, w, h).
        """
        # NOTE: consider using kornia BoundingBox
        # NOTE: consider having a separate function to convert the detections to a list of bounding boxes

        # https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/post_process.py#L446
        # box format is cxcywh
        # convert to xywh
        # bboxes[..., :2] -= bboxes[..., 2:] * 0.5  # in-place operation is not torch.compile()-friendly
        # TODO: replace using kornia BoundingBox
        cxcy, wh = boxes[..., :2], boxes[..., 2:]
        boxes_xy = concatenate([cxcy - wh * 0.5, wh], -1)

        # Get dynamic size from the input tensor itself
        sizes_wh = original_sizes[0].flip(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 2)

        boxes_xy = boxes_xy * sizes_wh
        scores = logits.sigmoid()  # RT-DETR was trained with focal loss. thus sigmoid is used instead of softmax

        # retrieve the boxes with the highest score for each class
        # https://github.com/lyuwenyu/RT-DETR/blob/b6bf0200b249a6e35b44e0308b6058f55b99696b/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py#L55-L62
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
        labels = mod(index, self.num_classes)
        index = index // self.num_classes
        boxes = boxes_xy.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes_xy.shape[-1]))

        all_boxes = concatenate([labels[..., None], scores[..., None], boxes], -1)

        if not self.confidence_filtering or self.confidence_threshold == 0:
            return all_boxes

        return self.box_filtering(all_boxes, self.confidence_threshold)
