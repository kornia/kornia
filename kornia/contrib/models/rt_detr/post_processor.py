from __future__ import annotations

# TODO:
import torch

from kornia.core import Module, Tensor, concatenate
from kornia.image.base import ImageSize


class DETRPostProcessor(Module):
    def __init__(self, confidence_threshold: float) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(self, logits: Tensor, boxes: Tensor, original_sizes: list[ImageSize]) -> list[Tensor]:
        """Post-process outputs from DETR.

        Args:
            logits: tensor with shape :math:`(N, Q, K)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries, :math:`K` is the number of classes.
            boxes: tensor with shape :math:`(N, Q, 4)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries.
            original_sizes: list of tuples, each tuple represent (img_height, img_width).

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

        sizes_wh = torch.empty(1, 1, 2, device=boxes.device, dtype=boxes.dtype)
        sizes_wh[..., 0] = original_sizes[0].width
        sizes_wh[..., 1] = original_sizes[0].height
        sizes_wh = sizes_wh.repeat(1, 1, 2)

        boxes_xy = boxes_xy * sizes_wh
        scores = logits.sigmoid()  # RT-DETR was trained with focal loss. thus sigmoid is used instead of softmax

        # the original code is slightly different
        # it allows 1 bounding box to have multiple classes (multi-label)
        scores, labels = scores.max(-1)

        detections: list[Tensor] = []
        for i in range(scores.shape[0]):
            mask = scores[i] >= self.confidence_threshold
            labels_i = labels[i, mask].unsqueeze(-1)
            scores_i = scores[i, mask].unsqueeze(-1)
            boxes_i = boxes_xy[i, mask]
            detections.append(concatenate([labels_i, scores_i, boxes_i], -1))

        return detections
