from typing import Any, ClassVar, Optional, Union

from kornia.core import Module, Tensor, rand, tensor
from kornia.core.mixin.onnx import ONNXExportMixin

__all__ = ["BoxFiltering"]


class BoxFiltering(Module, ONNXExportMixin):
    """Filter boxes according to the desired threshold.

    Args:
        confidence_threshold: an 0-d scalar that represents the desired threshold.
        classes_to_keep: a 1-d list of classes to keep. If None, keep all classes.
        filter_as_zero: whether to filter boxes as zero.
    """

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, -1, 6]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, 6]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[list[int]] = [5, 20, 6]

    def __init__(
        self,
        confidence_threshold: Optional[Union[Tensor, float]] = None,
        classes_to_keep: Optional[Union[Tensor, list[int]]] = None,
        filter_as_zero: bool = False,
    ) -> None:
        super().__init__()
        self.filter_as_zero = filter_as_zero
        self.classes_to_keep = None
        self.confidence_threshold = None
        if classes_to_keep is not None:
            self.classes_to_keep = classes_to_keep if isinstance(classes_to_keep, Tensor) else tensor(classes_to_keep)
        if confidence_threshold is not None:
            self.confidence_threshold = (
                confidence_threshold or confidence_threshold
                if isinstance(confidence_threshold, Tensor)
                else tensor(confidence_threshold)
            )

    def forward(
        self, boxes: Tensor, confidence_threshold: Optional[Tensor] = None, classes_to_keep: Optional[Tensor] = None
    ) -> Union[Tensor, list[Tensor]]:
        """Filter boxes according to the desired threshold.

        To be ONNX-friendly, the inputs for direct forwarding need to be all tensors.

        Args:
            boxes: [B, D, 6], where B is the batchsize,  D is the number of detections in the image,
                6 represent (class_id, confidence_score, x, y, w, h).
            confidence_threshold: an 0-d scalar that represents the desired threshold.
            classes_to_keep: a 1-d tensor of classes to keep. If None, keep all classes.

        Returns:
            Union[Tensor, List[Tensor]]
                If `filter_as_zero` is True, return a tensor of shape [D, 6], where D is the total number of
                detections as input.
                If `filter_as_zero` is False, return a list of tensors of shape [D, 6], where D is the number of
                valid detections for each element in the batch.
        """
        # Apply confidence filtering
        zero_tensor = tensor(0.0, device=boxes.device, dtype=boxes.dtype)
        confidence_threshold = (
            confidence_threshold or self.confidence_threshold or zero_tensor
        )  # If None, use 0 as threshold
        confidence_mask = boxes[:, :, 1] > confidence_threshold  # [B, D]

        # Apply class filtering
        classes_to_keep = classes_to_keep or self.classes_to_keep
        if classes_to_keep is not None:
            class_ids = boxes[:, :, 0:1]  # [B, D, 1]
            classes_to_keep = classes_to_keep.view(1, 1, -1)  # [1, 1, C] for broadcasting
            class_mask = (class_ids == classes_to_keep).any(dim=-1)  # [B, D]
        else:
            # If no class filtering is needed, just use a mask of all `True`
            class_mask = (confidence_mask * 0 + 1).bool()

        # Combine the confidence and class masks
        combined_mask = confidence_mask & class_mask  # [B, D]

        if self.filter_as_zero:
            filtered_boxes = boxes * combined_mask[:, :, None]
            return filtered_boxes

        filtered_boxes_list = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            mask = combined_mask[i]  # [D]
            valid_boxes = box[mask]
            filtered_boxes_list.append(valid_boxes)

        return filtered_boxes_list

    def _create_dummy_input(
        self, input_shape: list[int], pseudo_shape: Optional[list[int]] = None
    ) -> Union[tuple[Any, ...], Tensor]:
        pseudo_input = rand(
            *[
                ((self.ONNX_EXPORT_PSEUDO_SHAPE[i] if pseudo_shape is None else pseudo_shape[i]) if dim == -1 else dim)
                for i, dim in enumerate(input_shape)
            ]
        )
        if self.confidence_threshold is None:
            return pseudo_input, 0.1
        return pseudo_input
