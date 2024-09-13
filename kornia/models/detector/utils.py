from typing import Any, ClassVar, List, Optional, Tuple, Union

from kornia.core import Module, ONNXExportMixin, Tensor, rand

__all__ = ["BoxFiltering"]


class BoxFiltering(Module, ONNXExportMixin):
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = [-1, -1, 6]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[List[int]] = [5, 20, 6]

    def __init__(self, confidence_threshold: Optional[Tensor] = None, filter_as_zero: bool = False) -> None:
        super().__init__()
        self.filter_as_zero = filter_as_zero
        self.confidence_threshold = confidence_threshold

    def forward(self, boxes: Tensor, confidence_threshold: Optional[Tensor] = None) -> Union[Tensor, List[Tensor]]:
        """Filter boxes according to the desired threshold.

        Args:
            boxes: [B, D, 6], where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            if confidence_threshold is None:
                raise ValueError("`confidence_threshold` must be provided if not set in the constructor.")

        if self.filter_as_zero:
            box_mask = (boxes[:, :, 1] > confidence_threshold).unsqueeze(-1).expand_as(boxes)
            filtered_boxes = boxes * box_mask.float()
        else:
            filtered_boxes = []
            for i in range(boxes.shape[0]):
                box = boxes[i : i + 1][
                    (boxes[i : i + 1, :, 1] > confidence_threshold).unsqueeze(-1).expand_as(boxes[i : i + 1])
                ]
                filtered_boxes.append(box.view(1, -1, boxes.shape[-1]))

        return filtered_boxes

    def _create_dummy_input(self, input_shape: List[int]) -> Union[Tuple[Any, ...], Tensor]:
        pseudo_input = rand(
            *[(self.ONNX_EXPORT_PSEUDO_SHAPE[i] if dim == -1 else dim) for i, dim in enumerate(input_shape)]
        )
        if self.confidence_threshold is None:
            return pseudo_input, 0.1
        return pseudo_input
