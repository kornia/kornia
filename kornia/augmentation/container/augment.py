import warnings
from itertools import zip_longest
from typing import cast, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from kornia.augmentation.base import (
    _AugmentationBase,
    GeometricAugmentationBase2D,
    IntensityAugmentationBase2D,
    TensorWithTransformMat,
)
from kornia.constants import DataKey

from .base import SequentialBase
from .image import ImageSequential, ParamItem
from .patch import PatchSequential
from .utils import ApplyInverse
from .video import VideoSequential

__all__ = ["AugmentationSequential"]


class AugmentationSequential(ImageSequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_augmentation_sequential_5_1.png
        :width: 49 %
    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_augmentation_sequential_7_0.png
        :width: 49 %

    Args:
        *args: a list of kornia augmentation modules.
        data_keys: the input type sequential for applying augmentations.
            Accepts "input", "mask", "bbox", "bbox_xyxy", "bbox_xywh", "keypoints".
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.

    .. note::
        Mix augmentations (e.g. RandomMixUp, RandomCutMix) can only be working with "input" data key.
        It is not clear how to deal with the conversions of masks, bounding boxes and keypoints.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/data_augmentation_sequential.html>`__.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)
        >>> aug_list = AugmentationSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     data_keys=["input", "mask", "bbox", "keypoints"],
        ...     return_transform=False,
        ...     same_on_batch=False,
        ...     random_apply=10,
        ... )
        >>> out = aug_list(input, input, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
        >>> out_inv = aug_list.inverse(*out)
        >>> [o.shape for o in out_inv]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]

    Examples:
        This example demonstrates the integration of VideoSequential and AugmentationSequential.
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)[None]
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)[None]
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)[None]
        >>> aug_list = AugmentationSequential(
        ...     VideoSequential(
        ...         kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...     ),
        ...     data_keys=["input", "mask", "bbox", "keypoints"]
        ... )
        >>> out = aug_list(input, input, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 4, 2]), torch.Size([1, 2, 1, 2])]
    """

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        return_transform: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            return_transform=return_transform,
            keepdim=keepdim,
            random_apply=random_apply,
        )

        self.data_keys = [DataKey.get(inp) for inp in data_keys]

        if not all(in_type in DataKey for in_type in self.data_keys):
            raise AssertionError(f"`data_keys` must be in {DataKey}. Got {data_keys}.")

        if self.data_keys[0] != DataKey.INPUT:
            raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.contains_video_sequential: bool = False
        for arg in args:
            if isinstance(arg, PatchSequential) and not arg.is_intensity_only():
                warnings.warn("Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.")
            if isinstance(arg, VideoSequential):
                self.contains_video_sequential = True

    def inverse(  # type: ignore
        self,
        *args: torch.Tensor,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys`` is not set, use
        ``self.data_keys`` by default.
        """
        if data_keys is None:
            data_keys = cast(List[Union[str, int, DataKey]], self.data_keys)
        if len(args) != len(data_keys):
            raise AssertionError(
                "The number of inputs must align with the number of data_keys, "
                f"Got {len(args)} and {len(data_keys)}."
            )
        if params is None:
            if self._params is None:
                raise ValueError(
                    "No parameters available for inversing, please run a forward pass first "
                    "or passing valid params into this function."
                )
            params = self._params

        outputs = []
        for input, dcate in zip(args, data_keys):
            if dcate == DataKey.INPUT and isinstance(input, (tuple, list)):
                input, _ = input  # ignore the transformation matrix whilst inverse
            for (name, module), param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
                if isinstance(module, (_AugmentationBase, ImageSequential)):
                    param = params[name] if name in params else param
                else:
                    param = None
                if isinstance(module, IntensityAugmentationBase2D) and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, ImageSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, VideoSequential) and dcate not in [DataKey.INPUT, DataKey.MASK]:
                    batch_size: int = input.size(0)
                    input = input.view(-1, *input.shape[2:])
                    input = ApplyInverse.inverse_by_key(input, module, param, dcate)
                    input = input.view(batch_size, -1, *input.shape[1:])
                elif isinstance(module, PatchSequential):
                    raise NotImplementedError("Geometric involved PatchSequential is not supported.")
                elif isinstance(module, (GeometricAugmentationBase2D, ImageSequential)) and dcate in DataKey:
                    input = ApplyInverse.inverse_by_key(input, module, param, dcate)
                elif isinstance(module, (SequentialBase,)):
                    raise ValueError(f"Unsupported Sequential {module}.")
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")
            outputs.append(input)

        if len(outputs) == 1 and isinstance(outputs, (tuple, list)):
            return outputs[0]

        return outputs

    def __packup_output__(  # type: ignore
        self, output: List[TensorWithTransformMat], label: Optional[torch.Tensor] = None
    ) -> Union[
        TensorWithTransformMat,
        Tuple[TensorWithTransformMat, Optional[torch.Tensor]],
        List[TensorWithTransformMat],
        Tuple[List[TensorWithTransformMat], Optional[torch.Tensor]],
    ]:
        if len(output) == 1 and isinstance(output, (tuple, list)) and self.return_label:
            return output[0], label
        if len(output) == 1 and isinstance(output, (tuple, list)):
            return output[0]
        if self.return_label:
            return output, label
        return output

    def forward(  # type: ignore
        self,
        *args: TensorWithTransformMat,
        label: Optional[torch.Tensor] = None,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[
        TensorWithTransformMat,
        Tuple[TensorWithTransformMat, Optional[torch.Tensor]],
        List[TensorWithTransformMat],
        Tuple[List[TensorWithTransformMat], Optional[torch.Tensor]],
    ]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        if data_keys is None:
            data_keys = cast(List[Union[str, int, DataKey]], self.data_keys)
        else:
            data_keys = [DataKey.get(inp) for inp in data_keys]

        if len(args) != len(data_keys):
            raise AssertionError(
                f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."
            )

        if params is None:
            if DataKey.INPUT in data_keys:
                _input = args[data_keys.index(DataKey.INPUT)]
                if isinstance(_input, (tuple, list)):
                    inp = _input[0]
                else:
                    inp = _input
                if self.contains_video_sequential:
                    _, out_shape = self.autofill_dim(inp, dim_range=(3, 5))
                else:
                    _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
                params = self.forward_parameters(out_shape)
            else:
                raise ValueError("`params` must be provided whilst INPUT is not in data_keys.")

        outputs: List[TensorWithTransformMat] = [None] * len(data_keys)  # type: ignore
        if DataKey.INPUT in data_keys:
            idx = data_keys.index(DataKey.INPUT)
            out = super().forward(args[idx], label, params=params)
            if self.return_label:
                input, label = cast(Tuple[TensorWithTransformMat, torch.Tensor], out)
            else:
                input = cast(TensorWithTransformMat, out)
            outputs[idx] = input

        self.return_label = label is not None or self.contains_label_operations(params)

        for idx, (input, dcate, out) in enumerate(zip(args, data_keys, outputs)):
            if out is not None:
                continue
            for param in params:
                module = self.get_submodule(param.name)
                if dcate == DataKey.INPUT:
                    input, label = self.apply_to_input(input, label, module=module, param=param)
                elif isinstance(module, IntensityAugmentationBase2D) and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, ImageSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, VideoSequential) and dcate not in [DataKey.INPUT, DataKey.MASK]:
                    batch_size: int = input.size(0)
                    input = input.view(-1, *input.shape[2:])
                    input, label = ApplyInverse.apply_by_key(input, label, module, param, dcate)
                    input = input.view(batch_size, -1, *input.shape[1:])
                elif isinstance(module, PatchSequential):
                    raise NotImplementedError("Geometric involved PatchSequential is not supported.")
                elif isinstance(module, (GeometricAugmentationBase2D, ImageSequential,)) and dcate in DataKey:
                    input, label = ApplyInverse.apply_by_key(input, label, module, param, dcate)
                elif isinstance(module, (SequentialBase,)):
                    raise ValueError(f"Unsupported Sequential {module}.")
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")
            outputs[idx] = input

        return self.__packup_output__(outputs, label)
