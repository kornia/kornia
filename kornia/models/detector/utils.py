from typing import Optional

from kornia.core import Module


class BoxFiltering(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, boxes: Tensor, confidence_threshold: Tensor) -> Tensor:
        """Filter boxes according to the desired threshold.

        Args:
            boxes: [B, D, 6], where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
        """

        return boxes[(boxes[:, :, 1] > confidence_threshold).unsqueeze(-1).expand_as(boxes)].view(
            boxes.shape[0], -1, boxes.shape[-1]
        )
