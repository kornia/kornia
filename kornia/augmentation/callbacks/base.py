from typing import Dict, List, Optional, Union, cast

# NOTE: fix circular import
import kornia.augmentation as K
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
    ) -> None:
        """Called when `transform_inputs` begins."""
        ...

    def on_transform_inputs_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_inputs` ends."""
        ...

    def on_transform_masks_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_masks` begins."""
        ...

    def on_transform_masks_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_masks` ends."""
        ...

    def on_transform_classes_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_classes` begins."""
        ...

    def on_transform_classes_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_classes` ends."""
        ...

    def on_transform_boxes_start(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_boxes` begins."""
        ...

    def on_transform_boxes_end(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_boxes` ends."""
        ...

    def on_transform_keypoints_start(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_keypoints` begins."""
        ...

    def on_transform_keypoints_end(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `transform_keypoints` ends."""
        ...

    def on_inverse_inputs_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_input` begins."""
        ...

    def on_inverse_inputs_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_inputs` ends."""
        ...

    def on_inverse_masks_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_masks` begins."""
        ...

    def on_inverse_masks_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_masks` ends."""
        ...

    def on_inverse_classes_start(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_classes` begins."""
        ...

    def on_inverse_classes_end(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_classes` ends."""
        ...

    def on_inverse_boxes_start(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_boxes` begins."""
        ...

    def on_inverse_boxes_end(
        self,
        input: Boxes,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_boxes` ends."""
        ...

    def on_inverse_keypoints_start(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_keypoints` begins."""
        ...

    def on_inverse_keypoints_end(
        self,
        input: Keypoints,
        params: Dict[str, Tensor],
        module: object,
    ) -> None:
        """Called when `inverse_keypoints` ends."""
        ...

    def on_sequential_forward_start(
        self,
        *args: "K.container.data_types.DataType",
        module: "K.AugmentationSequential",
        params: List["K.container.params.ParamItem"],
        data_keys: Union[List[str], List[int], List[DataKey]],
    ) -> None:
        """Called when `forward` begins for `AugmentationSequential`."""
        ...

    def on_sequential_forward_end(
        self,
        *args: "K.container.data_types.DataType",
        module: "K.AugmentationSequential",
        params: List["K.container.params.ParamItem"],
        data_keys: Union[List[str], List[int], List[DataKey]],
    ) -> None:
        """Called when `forward` ends for `AugmentationSequential`."""
        ...

    def on_sequential_inverse_start(
        self,
        *args: "K.container.data_types.DataType",
        module: "K.AugmentationSequential",
        params: List["K.container.params.ParamItem"],
        data_keys: Union[List[str], List[int], List[DataKey]],
    ) -> None:
        """Called when `inverse` begins for `AugmentationSequential`."""
        ...

    def on_sequential_inverse_end(
        self,
        *args: "K.container.data_types.DataType",
        module: "K.AugmentationSequential",
        params: List["K.container.params.ParamItem"],
        data_keys: Union[List[str], List[int], List[DataKey]],
    ) -> None:
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
        self.batches_to_save = batches_to_save
        self.log_indices = log_indices
        self.data_keys = data_keys
        self.postprocessing = postprocessing
        self.num_to_log = num_to_log

    def _make_mask_data(self, mask: Tensor) -> Tensor:
        return mask

    def _make_bbox_data(self, bbox: Boxes) -> Tensor:
        return cast(Tensor, bbox.to_tensor("xyxy", as_padded_sequence=True))

    def _make_keypoints_data(self, keypoints: Keypoints) -> Tensor:
        return cast(Tensor, keypoints.to_tensor(as_padded_sequence=True))

    def _log_data(self, data: List[Tensor]) -> None:
        raise NotImplementedError

    def on_sequential_forward_end(
        self,
        *args: "K.container.data_types.DataType",
        module: "K.AugmentationSequential",
        params: List["K.container.params.ParamItem"],
        data_keys: Union[List[str], List[int], List[DataKey]],
    ) -> None:
        """Called when `forward` ends for `AugmentationSequential`."""
        output_data: List[Tensor] = []
        
        # Log all the indices
        if self.log_indices is None:
            self.log_indices = list(range(len(data_keys)))

        for i, (arg, data_key) in enumerate(zip(args, data_keys)):
            if i not in self.log_indices:
                continue

            postproc = None
            if self.postprocessing is not None:
                postproc = self.postprocessing[self.log_indices[i]]
            data = arg[:self.num_to_log]

            if postproc is not None:
                data = postproc(data)

            if data_key in [DataKey.INPUT]:
                output_data.append(cast(Tensor, data))
            if data_key in [DataKey.MASK]:
                output_data.append(self._make_mask_data(cast(Tensor, data)))
            if data_key in [DataKey.BBOX, DataKey.BBOX_XYWH, DataKey.BBOX_XYXY]:
                output_data.append(self._make_bbox_data(cast(Boxes, data)))
            if data_key in [DataKey.KEYPOINTS]:
                output_data.append(self._make_keypoints_data(cast(Keypoints, data)))

        self._log_data(output_data)
