import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation.base import TensorWithTransformMat, _BasicAugmentationBase
from kornia.augmentation.utils import _transform_input, _transform_output_shape, _validate_input_dtype
from kornia.constants import DataKey
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.testing import KORNIA_UNWRAP


class MixAugmentationBase(_BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized mix augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required.
    "apply_transform" will need to handle the probabilities internally.

    Args:
        p: probability for applying an augmentation. This param controls if to apply the augmentation for the batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
    """

    def __init__(self, p: float, p_batch: float, same_on_batch: bool = False, keepdim: bool = False) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        warnings.warn("`MixAugmentationBase` is deprecated. Please use `MixAugmentationBaseV2` instead.")

    def __check_batching__(self, input: TensorWithTransformMat):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                if len(mat.shape) != 3:
                    raise AssertionError('Input tensor is in batch mode ' 'but transformation matrix is not')
                if mat.shape[0] != inp.shape[0]:
                    raise AssertionError(
                        f"In batch dimension, input has {inp.shape[0]} but transformation matrix has {mat.shape[0]}"
                    )
            elif len(inp.shape) in (2, 3):
                if len(mat.shape) != 2:
                    raise AssertionError("Input tensor is in non-batch mode but transformation matrix is not")
            else:
                raise ValueError(f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}')

    def __unpack_input__(  # type: ignore
        self, input: TensorWithTransformMat
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def transform_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(  # type: ignore
        self, input: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def apply_func(  # type: ignore
        self, in_tensor: torch.Tensor, label: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        to_apply = params['batch_prob']

        # if no augmentation needed
        if torch.sum(to_apply) == 0:
            output = in_tensor
        # if all data needs to be augmented
        elif torch.sum(to_apply) == len(to_apply):
            output, label = self.apply_transform(in_tensor, label, params)
        else:
            raise ValueError(
                "Mix augmentations must be performed batch-wisely. Element-wise augmentation is not supported."
            )

        return output.type(in_tensor.dtype), label

    def forward(  # type: ignore
        self,
        input: TensorWithTransformMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[TensorWithTransformMat, torch.Tensor]:
        in_tensor, in_trans = self.__unpack_input__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        # If label is not provided, it would output the indices instead.
        if label is None:
            if isinstance(input, (tuple, list)):
                device = input[0].device
            else:
                device = input.device
            label = torch.arange(0, in_tensor.size(0), device=device, dtype=torch.long)
        if params is None:
            batch_shape = in_tensor.shape
            params = self.forward_parameters(batch_shape)
        self._params = params

        output, lab = self.apply_func(in_tensor, label, self._params)
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output  # type: ignore
        if in_trans is not None:
            return (output, in_trans), lab
        return output, lab


class MixAugmentationBaseV2(_BasicAugmentationBase):
    r"""MixAugmentationBase base class for customized mix augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required.
    "apply_transform" will need to handle the probabilities internally.

    Args:
        p: probability for applying an augmentation. This param controls if to apply the augmentation for the batch.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
          probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.
        data_keys: the input type sequential for applying augmentations.
            Accepts "input", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
    """

    def __init__(
        self,
        p: float,
        p_batch: float,
        same_on_batch: bool = False,
        keepdim: bool = False,
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
    ) -> None:
        super().__init__(p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        self.data_keys = [DataKey.get(inp) for inp in data_keys]

    def transform_tensor(self, input: Tensor) -> Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])
        return _transform_input(input)

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        # NOTE: apply_transform receives the whole tensor, but returns only altered elements.
        raise NotImplementedError

    def apply_non_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        # For the images where batch_prob == False.
        return input

    def transform_input(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        to_apply = params['batch_prob']
        ori_shape = input.shape
        in_tensor = self.transform_tensor(input)
        output = in_tensor
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform(in_tensor, params, flags)
        if sum(to_apply) != 0:
            applied = self.apply_transform(in_tensor, params, flags)
            output = self.apply_non_transform(in_tensor, params, flags)
            output = output.index_put((to_apply,), self.apply_non_transform(applied, params, flags))
        output = _transform_output_shape(output, ori_shape) if self.keepdim else output
        return output

    def transform_mask(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_mask(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_mask(input, params, flags)
        return output

    def transform_boxes(self, input: Union[Tensor, Boxes], params: Dict[str, Tensor], flags: Dict[str, Any]) -> Boxes:
        # input is BxNx4x2 or Boxes.
        if isinstance(input, Tensor):
            if not (len(input.shape) == 4 and input.shape[2:] == torch.Size([4, 2])):
                raise RuntimeError(f"Only BxNx4x2 tensor is supported. Got {input.shape}.")
            input = Boxes(input, False, mode="vertices_plus")
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_boxes(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_boxes(output, params, flags)
        return output

    def transform_keypoint(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_keypoint(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_keypoint(input, params, flags)
        return output

    def transform_class(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        to_apply = params['batch_prob']
        output = input
        if sum(to_apply) != len(to_apply):
            output = self.apply_non_transform_class(input, params, flags)
        if sum(to_apply) != 0:
            output = self.apply_transform_class(input, params, flags)
        return output

    def apply_non_transform_mask(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_transform_mask(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_non_transform_boxes(self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Boxes:
        return input

    def apply_transform_boxes(self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Boxes:
        raise NotImplementedError

    def apply_non_transform_keypoint(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return input

    def apply_transform_keypoint(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def apply_non_transform_class(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return input

    def apply_transform_class(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def forward(  # type: ignore
        self,
        *input: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[Tensor, List[Tensor]]:
        keys: List[DataKey]
        if data_keys is None:
            keys = self.data_keys
        else:
            keys = [DataKey.get(inp) for inp in data_keys]

        if params is None:
            in_tensor_idx: int = keys.index(DataKey.INPUT)
            in_tensor: Tensor = input[in_tensor_idx]
            in_tensor = self.transform_tensor(in_tensor)
            self._params = self.forward_parameters(in_tensor.shape)
            self._params.update({"device": in_tensor.device, "dtype": in_tensor.dtype})
        else:
            self._params = params

        outputs = []
        for dcate, _input in zip(keys, input):
            output: Tensor
            if dcate == DataKey.INPUT:
                output = self.transform_input(_input, self._params, self.flags)
            elif dcate == DataKey.MASK:
                output = self.transform_mask(_input, self._params, self.flags)
            elif dcate == DataKey.BBOX:
                box = Boxes.from_tensor(_input, mode="vertices", validate_boxes=False)
                box = self.transform_boxes(box, self._params, self.flags)
                output = KORNIA_UNWRAP(box.to_tensor("vertices"), Tensor)
            elif dcate == DataKey.BBOX_XYXY:
                box = Boxes.from_tensor(_input, mode="xyxy", validate_boxes=False)
                box = self.transform_boxes(box, self._params, self.flags)
                output = KORNIA_UNWRAP(box.to_tensor("xyxy"), Tensor)
            elif dcate == DataKey.BBOX_XYWH:
                box = Boxes.from_tensor(_input, mode="xywh", validate_boxes=False)
                box = self.transform_boxes(box, self._params, self.flags)
                output = KORNIA_UNWRAP(box.to_tensor("xywh"), Tensor)
            elif dcate == DataKey.KEYPOINTS:
                output = self.transform_keypoint(_input, self._params, self.flags)
            elif dcate == DataKey.CLASS:
                output = self.transform_class(_input, self._params, self.flags)
            else:
                raise NotImplementedError
            outputs.append(output)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @torch.jit.ignore
    def inverse(self, **kwargs):
        raise RuntimeError(f"Inverse for {self.__class__.__name__} is not supported.")

    @property
    def transform_matrix(self):
        raise RuntimeError(f"Transformation matrices for {self.__class__.__name__} is not supported.")
