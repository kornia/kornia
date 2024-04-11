from typing import Dict, List, Optional, Union

# NOTE: fix circular import
import kornia.augmentation as K
# .data_types import DataType
# from kornia.augmentation.container.params import ParamItem
from kornia.constants import DataKey
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

__all__ = [
    "AugmentationCallbackBase",
]


class AugmentationCallbackBase(Module):
    """A Meta Callback base class."""

    def on_transform_inputs_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_inputs` begins."""
        ...

    def on_transform_inputs_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_inputs` ends."""
        ...

    def on_transform_masks_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_masks` begins."""
        ...

    def on_transform_masks_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_masks` ends."""
        ...

    def on_transform_classes_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_classes` begins."""
        ...

    def on_transform_classes_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_classes` ends."""
        ...

    def on_transform_boxes_start(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_boxes` begins."""
        ...

    def on_transform_boxes_end(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_boxes` ends."""
        ...

    def on_transform_keypoints_start(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_keypoints` begins."""
        ...

    def on_transform_keypoints_end(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `transform_keypoints` ends."""
        ...

    def on_inverse_inputs_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_input` begins."""
        ...

    def on_inverse_inputs_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_inputs` ends."""
        ...

    def on_inverse_masks_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_masks` begins."""
        ...

    def on_inverse_masks_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_masks` ends."""
        ...

    def on_inverse_classes_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_classes` begins."""
        ...

    def on_inverse_classes_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_classes` ends."""
        ...

    def on_inverse_boxes_start(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_boxes` begins."""
        ...

    def on_inverse_boxes_end(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_boxes` ends."""
        ...

    def on_inverse_keypoints_start(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_keypoints` begins."""
        ...

    def on_inverse_keypoints_end(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ):
        """Called when `inverse_keypoints` ends."""
        ...

    def on_sequential_forward_start(
        self,
        *args: Union["K.container.data_types.DataType", Dict[str, "K.container.data_types.DataType"]],
        params: Optional[List["K.container.params.DataType"]] = None,
        data_keys: Optional[Union[List[str], List[int], List["K.container.data_types.DataType"]]] = None,
        module: object,
    ):
        """Called when `forward` begins for `AugmentationSequential`."""
        ...

    def on_sequential_forward_end(
        self,
        *args: Union["K.container.data_types.DataType", Dict[str, "K.container.data_types.DataType"]],
        params: Optional[List["K.container.params.DataType"]] = None,
        data_keys: Optional[Union[List[str], List[int], List["K.container.data_types.DataType"]]] = None,
        module: object,
    ):
        """Called when `forward` ends for `AugmentationSequential`."""
        ...

    def on_sequential_inverse_start(
        self,
        *args: Union["K.container.data_types.DataType", Dict[str, "K.container.data_types.DataType"]],
        params: Optional[List["K.container.params.DataType"]] = None,
        data_keys: Optional[Union[List[str], List[int], List["K.container.data_types.DataType"]]] = None,
        module: object,
    ):
        """Called when `inverse` begins for `AugmentationSequential`."""
        ...

    def on_sequential_inverse_end(
        self,
        *args: Union["K.container.data_types.DataType", Dict[str, "K.container.data_types.DataType"]],
        params: Optional[List["K.container.params.DataType"]] = None,
        data_keys: Optional[Union[List[str], List[int], List["K.container.data_types.DataType"]]] = None,
        module: object,
    ):
        """Called when `inverse` ends for `AugmentationSequential`."""
        ...


class AugmentationCallback(AugmentationCallbackBase):
    """Logging images for `AugmentationSequential`.

    Args:
        batches_to_save: the number of batches to be logged. -1 is to save all batches.
        num_to_log: number of images to log in a batch.
        log_indices: only selected input types are logged. If `log_indices=[0, 2]` and
                     `data_keys=["input", "bbox", "mask"]`, only the images and masks
                     will be logged.
        data_keys: the input type sequential. Accepts "input", "image", "mask",
                   "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
        postprocessing: add postprocessing for images if needed. If not None, the length
                       must match `data_keys`.
    """

    def __init__(
        self,
        batches_to_save: int = 10,
        num_to_log: int = 4,
        log_indices: Optional[List[int]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
        postprocessing: Optional[List[Optional[Module]]] = None,
    ):
        super().__init__()
        self.batches_to_log = batches_to_log
        self.log_indices = log_indices
        self.data_keys = data_keys
        self.postprocessing = postprocessing
        self.num_to_log = num_to_log

    def _make_mask_data(self, mask: Tensor):
        raise NotImplementedError

    def _make_bbox_data(self, bbox: Tensor):
        raise NotImplementedError

    def _log_data(self, data: SequenceDataType):
        raise NotImplementedError

    def on_sequential_forward_end(
        self,
        *args: Union[DataType, Dict[str, DataType]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ):
        """Called when `forward` ends for `AugmentationSequential`."""
        image_data = None
        output_data = []
        for i, (arg, data_key) in enumerate(zip(args, data_keys)):
            if i not in self.log_indices:
                continue

            postproc = self.postprocessing[self.log_indices[i]]
            data = arg[: self.num_to_log]
            if postproc is not None:
                data = postproc(data)
            if data_key in [DataKey.INPUT]:
                data = data
            if data_key in [DataKey.MASK]:
                data = self._make_mask_data(data)
            if data_key in [DataKey.BBOX, DataKey.BBOX_XYWH, DataKey.BBOX_XYXY]:
                data = self._make_bbox_data(data)

            output_data.append(data)

        self._log_data(output_data)
