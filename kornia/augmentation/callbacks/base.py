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
