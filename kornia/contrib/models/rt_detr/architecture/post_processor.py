from __future__ import annotations

from kornia.core import Module, Tensor, concatenate, tensor


class DETRPostProcessor(Module):
    def __init__(self, confidence_threshold: float) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(self, logits: Tensor, boxes: Tensor, height: int, width: int) -> list[Tensor]:
        # https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/post_process.py#L446
        # box format is cxcywh
        # convert to xywh
        # bboxes[..., :2] -= bboxes[..., 2:] * 0.5  # in-place operation is not torch.compile()-friendly
        cxcy = boxes[..., :2]
        wh = boxes[..., 2:]
        boxes = concatenate([cxcy - wh * 0.5, wh], -1)

        boxes = boxes * tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype).view(1, 1, 4)
        scores = logits.sigmoid()  # RT-DETR was trained with focal loss. thus sigmoid is used instead of softmax

        # the original code is slightly different
        # it allows 1 bounding box to have multiple classes (multi-label)
        scores, labels = scores.max(-1)
        detections = []
        for i in range(scores.shape[0]):
            mask = scores[i] >= self.confidence_threshold
            labels_i = labels[i, mask].unsqueeze(-1)
            scores_i = scores[i, mask].unsqueeze(-1)
            boxes_i = boxes[i, mask]
            detections.append(concatenate([labels_i, scores_i, boxes_i], -1))
        return detections
