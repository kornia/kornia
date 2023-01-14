import warnings
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import (
    AugmentationBase3D,
    GeometricAugmentationBase2D,
    IntensityAugmentationBase2D,
    RandomErasing,
)
from kornia.augmentation._2d.mix.base import MixAugmentationBaseV2
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.base import SequentialBase
from kornia.augmentation.container.image import ImageSequential, ParamItem
from kornia.augmentation.container.patch import PatchSequential
from kornia.augmentation.container.utils import ApplyInverse
from kornia.augmentation.container.video import VideoSequential
from kornia.constants import DataKey, Resample
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.testing import KORNIA_CHECK_IS_LIST_OF_TENSOR
from kornia.utils import eye_like

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
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.
        extra_args: to control the behaviour for each datakeys. By default, masks are handled
            by nearest interpolation strategies.

    .. note::
        Mix augmentations (e.g. RandomMixUp, RandomCutMix) can only be working with "input" data key.
        It is not clear how to deal with the conversions of masks, bounding boxes and keypoints.

    .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/
        latest/data_augmentation_sequential.html>`__.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> mask = torch.ones(2, 3, 5, 6)
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)
        >>> aug_list = AugmentationSequential(
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     data_keys=["input", "mask", "bbox", "keypoints"],
        ...     same_on_batch=False,
        ...     random_apply=10,
        ... )
        >>> out = aug_list(input, mask, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]
        >>> # apply the exact augmentation again.
        >>> out_rep = aug_list(input, mask, bbox, points, params=aug_list._params)
        >>> [(o == o_rep).all() for o, o_rep in zip(out, out_rep)]
        [tensor(True), tensor(True), tensor(True), tensor(True)]
        >>> # inverse the augmentations
        >>> out_inv = aug_list.inverse(*out)
        >>> [o.shape for o in out_inv]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 4, 2]), torch.Size([2, 1, 2])]

    This example demonstrates the integration of VideoSequential and AugmentationSequential.

        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)[None]
        >>> mask = torch.ones(2, 3, 5, 6)[None]
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)[None]
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)[None]
        >>> aug_list = AugmentationSequential(
        ...     VideoSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...     ),
        ...     data_keys=["input", "mask", "bbox", "keypoints"]
        ... )
        >>> out = aug_list(input, mask, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 4, 2]), torch.Size([1, 2, 1, 2])]

    Perform ``OneOf`` transformation with ``random_apply=1`` and ``random_apply_weights`` in ``AugmentationSequential``.

        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)[None]
        >>> mask = torch.ones(2, 3, 5, 6)[None]
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, -1, -1)[None]
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)[None]
        >>> aug_list = AugmentationSequential(
        ...     VideoSequential(
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...     ),
        ...     VideoSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     ),
        ...     data_keys=["input", "mask", "bbox", "keypoints"],
        ...     random_apply=1,
        ...     random_apply_weights=[0.5, 0.3]
        ... )
        >>> out = aug_list(input, mask, bbox, points)
        >>> [o.shape for o in out]
        [torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 4, 2]), torch.Size([1, 2, 1, 2])]
    """

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        extra_args: Dict[DataKey, Dict[str, Any]] = {DataKey.MASK: dict(resample=Resample.NEAREST, align_corners=True)},
    ) -> None:
        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
            random_apply_weights=random_apply_weights,
        )

        self.data_keys = [DataKey.get(inp) for inp in data_keys]

        if not all(in_type in DataKey for in_type in self.data_keys):
            raise AssertionError(f"`data_keys` must be in {DataKey}. Got {data_keys}.")

        if self.data_keys[0] != DataKey.INPUT:
            raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.contains_video_sequential: bool = False
        self.contains_3d_augmentation: bool = False
        for arg in args:
            if isinstance(arg, PatchSequential) and not arg.is_intensity_only():
                warnings.warn("Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.")
            if isinstance(arg, VideoSequential):
                self.contains_video_sequential = True
            # NOTE: only for images are supported for 3D.
            if isinstance(arg, AugmentationBase3D):
                self.contains_3d_augmentation = True
        self._transform_matrix: Optional[Tensor] = None
        self.extra_args = extra_args

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        if self.contains_3d_augmentation:
            return eye_like(4, input)
        else:
            return eye_like(3, input)

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        return self._transform_matrix

    def inverse(  # type: ignore[override]
        self,
        *args: Tensor,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[Tensor, List[Tensor], List[Union[Tensor, List[Tensor]]]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys`` is not set, use
        ``self.data_keys`` by default.
        """
        if data_keys is None:
            _data_keys = self.data_keys
        else:
            _data_keys = [DataKey.get(inp) for inp in data_keys]

        self._validate_args_datakeys(*args, data_keys=_data_keys)

        args = self._arguments_preproc(*args, data_keys=_data_keys)

        if params is None:
            if self._params is None:
                raise ValueError(
                    "No parameters available for inversing, please run a forward pass first "
                    "or passing valid params into this function."
                )
            params = self._params

        outputs: List[Optional[Tensor]] = [None] * len(_data_keys)
        for idx, (arg, dcate) in enumerate(zip(args, _data_keys)):

            if dcate in self.extra_args:
                extra_args = self.extra_args[dcate]
            else:
                extra_args = {}

            if dcate == DataKey.INPUT and isinstance(arg, (tuple, list)):
                input, _ = arg  # ignore the transformation matrix whilst inverse
            # Using tensors straight-away
            elif isinstance(arg, (Boxes,)):
                input = arg.data  # all boxes are in (B, N, 4, 2) format now.
            else:
                input = arg
            for (name, module), _param in zip_longest(list(self.get_forward_sequence(params))[::-1], params[::-1]):
                if isinstance(module, (_AugmentationBase, ImageSequential)):
                    _mb = [p for p in params if name in p]
                    if len(_mb) > 0:
                        param = _mb[0]
                    elif isinstance(_param, ParamItem):
                        param = _param
                    else:
                        param = None
                else:
                    param = None

                if (
                    isinstance(module, IntensityAugmentationBase2D)
                    and dcate in DataKey
                    and not isinstance(module, RandomErasing)
                ):
                    pass  # Do nothing
                elif isinstance(module, ImageSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, VideoSequential) and dcate not in [DataKey.INPUT, DataKey.MASK]:
                    batch_size: int = input.size(0)
                    input = input.view(-1, *input.shape[2:])
                    input = ApplyInverse.inverse_by_key(input, module, param, dcate, extra_args=extra_args)
                    input = input.view(batch_size, -1, *input.shape[1:])
                elif isinstance(module, PatchSequential):
                    raise NotImplementedError("Geometric involved PatchSequential is not supported.")
                elif isinstance(module, (AugmentationSequential)) and dcate in DataKey:
                    # AugmentationSequential shall not take the extra_args arguments.
                    input = ApplyInverse.inverse_by_key(input, module, param, dcate)
                elif (
                    isinstance(module, (GeometricAugmentationBase2D, ImageSequential, RandomErasing))
                    and dcate in DataKey
                ):
                    input = ApplyInverse.inverse_by_key(input, module, param, dcate, extra_args=extra_args)
                elif isinstance(module, (SequentialBase,)):
                    raise ValueError(f"Unsupported Sequential {module}.")
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")
            if isinstance(arg, (Boxes,)):
                arg._data = input
                outputs[idx] = arg.to_tensor()
            else:
                outputs[idx] = input

        _outputs = [i for i in outputs if isinstance(i, Tensor) or KORNIA_CHECK_IS_LIST_OF_TENSOR(i)]

        if len(_outputs) == 1 and isinstance(_outputs, list):
            return _outputs[0]

        return _outputs

    def __packup_output__(  # type: ignore[override]
        self, output: List[Union[Tensor, List[Tensor]]], label: Optional[Tensor] = None
    ) -> Union[
        Tensor,
        List[Tensor],
        List[Union[Tensor, List[Tensor]]],
        Tuple[Union[Tensor, List[Tensor], List[Union[Tensor, List[Tensor]]]], Optional[Tensor]],
    ]:

        _out: Union[Tensor, List[Tensor], List[Union[Tensor, List[Tensor]]]]

        if len(output) == 1 and isinstance(output, list):
            _out = output[0]
        else:
            _out = output

        if self.return_label:
            return _out, label

        return _out

    def _validate_args_datakeys(self, *args: Tensor, data_keys: List[DataKey]):
        if len(args) != len(data_keys):
            raise AssertionError(
                f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."
            )
        # TODO: validate args batching, and its consistency

    def _arguments_preproc(self, *args: Tensor, data_keys: List[DataKey]):
        inp: List[Any] = []
        for arg, dcate in zip(args, data_keys):
            if DataKey.get(dcate) in [DataKey.INPUT, DataKey.MASK, DataKey.KEYPOINTS]:
                inp.append(arg)
            elif DataKey.get(dcate) in [DataKey.BBOX, DataKey.BBOX_XYXY, DataKey.BBOX_XYWH]:
                if DataKey.get(dcate) in [DataKey.BBOX]:
                    mode = "vertices_plus"
                elif DataKey.get(dcate) in [DataKey.BBOX_XYXY]:
                    mode = "xyxy"
                elif DataKey.get(dcate) in [DataKey.BBOX_XYWH]:
                    mode = "xywh"
                else:
                    raise ValueError(f"Unsupported mode `{DataKey.get(dcate).name}`.")
                inp.append(Boxes.from_tensor(arg, mode=mode))
            else:
                raise NotImplementedError(f"input type of {dcate} is not implemented.")
        return inp

    def forward(  # type: ignore[override]
        self,
        *args: Tensor,
        label: Optional[Tensor] = None,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[
        Tensor,
        List[Tensor],
        List[Union[Tensor, List[Tensor]]],
        Tuple[Union[Tensor, List[Tensor], List[Union[Tensor, List[Tensor]]]], Optional[Tensor]],
    ]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        if data_keys is None:
            _data_keys = self.data_keys
        else:
            _data_keys = [DataKey.get(inp) for inp in data_keys]

        self._validate_args_datakeys(*args, data_keys=_data_keys)

        args = self._arguments_preproc(*args, data_keys=_data_keys)

        if params is None:
            # image data must exist if params is not provided.
            if DataKey.INPUT in _data_keys:
                inp = args[_data_keys.index(DataKey.INPUT)]
                if isinstance(inp, (tuple, list)):
                    raise ValueError(f"`INPUT` should be a tensor but `{type(inp)}` received.")
                # A video input shall be BCDHW while an image input shall be BCHW
                if self.contains_video_sequential or self.contains_3d_augmentation:
                    _, out_shape = self.autofill_dim(inp, dim_range=(3, 5))
                else:
                    _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
                params = self.forward_parameters(out_shape)
            else:
                raise ValueError("`params` must be provided whilst INPUT is not in data_keys.")

        outputs: List[Optional[Tensor]] = [None] * len(_data_keys)

        self.return_label = self.return_label or label is not None or self.contains_label_operations(params)

        for idx, (arg, dcate) in enumerate(zip(args, _data_keys)):
            # Forward the param to all input data keys
            if dcate in self.extra_args:
                extra_args = self.extra_args[dcate]
            else:
                extra_args = {}

            if dcate == DataKey.INPUT:
                _inp = args[idx]

                _out = super().forward(_inp, label, params=params, extra_args=extra_args)
                self._transform_matrix = self.get_transformation_matrix(_inp, params=params)

                if self.return_label and isinstance(_out, tuple):
                    _input, label = _out
                elif isinstance(_out, Tensor):
                    _input = _out

                outputs[idx] = _input
                # NOTE: Skip the rest here.
                continue

            # Using tensors straight-away
            if isinstance(arg, (Boxes,)):
                input = arg.data  # all boxes are in (B, N, 4, 2) format now.
            else:
                input = arg

            for param in params:
                module = self.get_submodule(param.name)
                if (
                    isinstance(module, IntensityAugmentationBase2D)
                    and dcate in DataKey
                    and not isinstance(module, RandomErasing)
                ):
                    pass  # Do nothing
                elif isinstance(module, ImageSequential) and module.is_intensity_only() and dcate in DataKey:
                    pass  # Do nothing
                elif isinstance(module, VideoSequential) and dcate not in [DataKey.INPUT, DataKey.MASK]:
                    batch_size: int = input.size(0)
                    input = input.view(-1, *input.shape[2:])
                    input, label = ApplyInverse.apply_by_key(input, label, module, param, dcate, extra_args=extra_args)
                    input = input.view(batch_size, -1, *input.shape[1:])
                elif isinstance(module, PatchSequential):
                    raise NotImplementedError("Geometric involved PatchSequential is not supported.")
                elif (
                    isinstance(module, (GeometricAugmentationBase2D, ImageSequential, RandomErasing))
                    and dcate in DataKey
                ):
                    input, label = ApplyInverse.apply_by_key(input, label, module, param, dcate, extra_args=extra_args)
                elif isinstance(module, MixAugmentationBaseV2):
                    if dcate in [DataKey.BBOX_XYXY, DataKey.BBOX_XYWH]:
                        dcate = DataKey.BBOX
                    input = module(input, params=param.data, data_keys=[dcate])
                elif isinstance(module, (SequentialBase,)):
                    raise ValueError(f"Unsupported Sequential {module}.")
                else:
                    raise NotImplementedError(f"data_key {dcate} is not implemented for {module}.")

            if isinstance(arg, (Boxes,)):
                arg._data = input
                outputs[idx] = arg.to_tensor()
            else:
                outputs[idx] = input
        _outputs = [i for i in outputs if isinstance(i, Tensor) or KORNIA_CHECK_IS_LIST_OF_TENSOR(i)]

        return self.__packup_output__(_outputs, label)
