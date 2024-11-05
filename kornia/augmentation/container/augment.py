import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.augmentation._3d.base import AugmentationBase3D, RigidAffineAugmentationBase3D
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey, Resample
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes, VideoBoxes
from kornia.geometry.keypoints import Keypoints, VideoKeypoints
from kornia.utils import eye_like, is_autocast_enabled

from .base import TransformMatrixMinIn
from .image import ImageSequential
from .ops import AugmentationSequentialOps, DataType
from .params import ParamItem
from .patch import PatchSequential
from .video import VideoSequential

__all__ = ["AugmentationSequential"]

_BOXES_OPTIONS = {DataKey.BBOX, DataKey.BBOX_XYXY, DataKey.BBOX_XYWH}
_KEYPOINTS_OPTIONS = {DataKey.KEYPOINTS}
_IMG_OPTIONS = {DataKey.INPUT, DataKey.IMAGE}
_MSK_OPTIONS = {DataKey.MASK}
_CLS_OPTIONS = {DataKey.CLASS, DataKey.LABEL}

MaskDataType = Union[Tensor, List[Tensor]]


class AugmentationSequential(TransformMatrixMinIn, ImageSequential):
    r"""AugmentationSequential for handling multiple input types like inputs, masks, keypoints at once.

    .. image:: _static/img/AugmentationSequential.png

    Args:
        *args: a list of kornia augmentation modules.

        data_keys: the input type sequential for applying augmentations. Accepts "input", "image", "mask",
                   "bbox", "bbox_xyxy", "bbox_xywh", "keypoints", "class", "label".

        same_on_batch: apply the same transformation across the batch. If None, it will not overwrite the function-wise
                       settings.

        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).
                 If None, it will not overwrite the function-wise settings.

        random_apply: randomly select a sublist (order agnostic) of args to apply transformation.
                      If int, a fixed number of transformations will be selected.
                      If (a,), x number of transformations (a <= x <= len(args)) will be selected.
                      If (a, b), x number of transformations (a <= x <= b) will be selected.
                      If True, the whole list of args will be processed as a sequence in a random order.
                      If False, the whole list of args will be processed as a sequence in original order.

        transformation_matrix_mode: computation mode for the chained transformation matrix, via `.transform_matrix`
                                    attribute.
                                    If `silent`, transformation matrix will be computed silently and the non-rigid
                                    modules will be ignored as identity transformations.
                                    If `rigid`, transformation matrix will be computed silently and the non-rigid
                                    modules will trigger errors.
                                    If `skip`, transformation matrix will be totally ignored.

        extra_args: to control the behaviour for each datakeys. By default, masks are handled by nearest interpolation
                    strategies.

    .. note::
        Mix augmentations (e.g. RandomMixUp, RandomCutMix) can only be working with "input"/"image" data key.
        It is not clear how to deal with the conversions of masks, bounding boxes and keypoints.

    .. note::
        See a working example `here <https://kornia.github.io/tutorials/nbs/data_augmentation_sequential.html>`__.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> mask = torch.ones(2, 3, 5, 6)
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, 1, -1, -1)
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
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 1, 4, 2]), torch.Size([2, 1, 2])]
        >>> # apply the exact augmentation again.
        >>> out_rep = aug_list(input, mask, bbox, points, params=aug_list._params)
        >>> [(o == o_rep).all() for o, o_rep in zip(out, out_rep)]
        [tensor(True), tensor(True), tensor(True), tensor(True)]
        >>> # inverse the augmentations
        >>> out_inv = aug_list.inverse(*out)
        >>> [o.shape for o in out_inv]
        [torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 5, 6]), torch.Size([2, 1, 4, 2]), torch.Size([2, 1, 2])]

    This example demonstrates the integration of VideoSequential and AugmentationSequential.

        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)[None]
        >>> mask = torch.ones(2, 3, 5, 6)[None]
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, 1, -1, -1)[None]
        >>> points = torch.tensor([[[1., 1.]]]).expand(2, -1, -1)[None]
        >>> aug_list = AugmentationSequential(
        ...     VideoSequential(
        ...         kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...         kornia.augmentation.RandomAffine(360, p=1.0),
        ...     ),
        ...     data_keys=["input", "mask", "bbox", "keypoints"]
        ... )
        >>> out = aug_list(input, mask, bbox, points)
        >>> [o.shape for o in out]  # doctest: +ELLIPSIS
        [torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 3, 5, 6]), ...([1, 2, 1, 4, 2]), torch.Size([1, 2, 1, 2])]

    Perform ``OneOf`` transformation with ``random_apply=1`` and ``random_apply_weights``
    in ``AugmentationSequential``.

        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)[None]
        >>> mask = torch.ones(2, 3, 5, 6)[None]
        >>> bbox = torch.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]]).expand(2, 1, -1, -1)[None]
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
        >>> [o.shape for o in out]  # doctest: +ELLIPSIS
        [torch.Size([1, 2, 3, 5, 6]), torch.Size([1, 2, 3, 5, 6]), ...([1, 2, 1, 4, 2]), torch.Size([1, 2, 1, 2])]

    This example shows how to use a list of masks and boxes within AugmentationSequential

        >>> import kornia.augmentation as K
        >>> input = torch.randn(2, 3, 256, 256)
        >>> mask = [torch.ones(1, 3, 256, 256), torch.ones(1, 2, 256, 256)]
        >>> bbox = [
        ...    torch.tensor([[28.0, 53.0, 143.0, 164.0], [254.0, 158.0, 364.0, 290.0], [307.0, 204.0, 413.0, 350.0]]),
        ...    torch.tensor([[254.0, 158.0, 364.0, 290.0], [307.0, 204.0, 413.0, 350.0]])
        ... ]
        >>> bbox = [Boxes.from_tensor(i).data for i in bbox]

        >>> aug_list = K.AugmentationSequential(
        ...    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...    K.RandomHorizontalFlip(p=1.0),
        ...    K.ImageSequential(K.RandomHorizontalFlip(p=1.0)),
        ...    K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)),
        ...    data_keys=["input", "mask", "bbox"],
        ...    same_on_batch=False,
        ...    random_apply=10,
        ... )
        >>> out = aug_list(input, mask, bbox)

    How to use a dictionary as input with AugmentationSequential? The dictionary keys that start with
    one of the available datakeys will be augmented accordingly. Otherwise, the dictionary item is passed
    without any augmentation.

        >>> import kornia.augmentation as K
        >>> img = torch.randn(1, 3, 256, 256)
        >>> mask = [torch.ones(1, 3, 256, 256), torch.ones(1, 2, 256, 256)]
        >>> bbox = [
        ...    torch.tensor([[28.0, 53.0, 143.0, 164.0], [254.0, 158.0, 364.0, 290.0], [307.0, 204.0, 413.0, 350.0]]),
        ...    torch.tensor([[254.0, 158.0, 364.0, 290.0], [307.0, 204.0, 413.0, 350.0]])
        ... ]
        >>> bbox = [Boxes.from_tensor(i).data for i in bbox]
        >>> aug_dict = K.AugmentationSequential(
        ...    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...    K.RandomHorizontalFlip(p=1.0),
        ...    K.ImageSequential(K.RandomHorizontalFlip(p=1.0)),
        ...    K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)),
        ...    data_keys=None,
        ...    same_on_batch=False,
        ...    random_apply=10,
        ... )
        >>> data = {'image': img, 'mask': mask[0], 'mask-b': mask[1], 'bbox': bbox[0], 'bbox-other':bbox[1]}
        >>> out = aug_dict(data)
        >>> out.keys()
        dict_keys(['image', 'mask', 'mask-b', 'bbox', 'bbox-other'])
    """

    input_dtype = None
    mask_dtype = None

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        transformation_matrix_mode: str = "silent",
        extra_args: Dict[DataKey, Dict[str, Any]] = {
            DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
        },
    ) -> None:
        self._transform_matrix: Optional[Tensor]
        self._transform_matrices: List[Optional[Tensor]] = []

        super().__init__(
            *args,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
            random_apply=random_apply,
            random_apply_weights=random_apply_weights,
        )

        self._parse_transformation_matrix_mode(transformation_matrix_mode)

        self._valid_ops_for_transform_computation: Tuple[Any, ...] = (
            RigidAffineAugmentationBase2D,
            RigidAffineAugmentationBase3D,
            AugmentationSequential,
        )

        self.data_keys: Optional[List[DataKey]]
        if data_keys is not None:
            self.data_keys = [DataKey.get(inp) for inp in data_keys]
        else:
            self.data_keys = data_keys

        if self.data_keys:
            if any(in_type not in DataKey for in_type in self.data_keys):
                raise AssertionError(f"`data_keys` must be in {DataKey}. Got {self.data_keys}.")

            if self.data_keys[0] != DataKey.INPUT:
                raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.transform_op = AugmentationSequentialOps(self.data_keys)

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
        self._transform_matrix = None
        self.extra_args = extra_args

    def clear_state(self) -> None:
        self._reset_transform_matrix_state()
        return super().clear_state()

    def _update_transform_matrix_for_valid_op(self, module: Module) -> None:
        self._transform_matrices.append(module.transform_matrix)

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        if self.contains_3d_augmentation:
            return eye_like(4, input)
        else:
            return eye_like(3, input)

    def inverse(  # type: ignore[override]
        self,
        *args: Union[DataType, Dict[str, DataType]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, List[DataType], Dict[str, DataType]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys`` is not set, use
        ``self.data_keys`` by default.
        """
        original_keys = None
        if len(args) == 1 and isinstance(args[0], dict):
            original_keys, data_keys, args, invalid_data = self._preproc_dict_data(args[0])

        # args here should already be `DataType`
        # NOTE: how to right type to: unpacked args <-> tuple of args to unpack
        # issue with `self._preproc_dict_data` return args type

        self.transform_op.data_keys = self.transform_op.preproc_datakeys(data_keys)

        self._validate_args_datakeys(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        in_args = self._arguments_preproc(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        if params is None:
            if self._params is None:
                raise ValueError(
                    "No parameters available for inversing, please run a forward pass first "
                    "or passing valid params into this function."
                )
            params = self._params

        outputs: List[DataType] = in_args
        for param in params[::-1]:
            module = self.get_submodule(param.name)
            outputs = self.transform_op.inverse(  # type: ignore
                *outputs, module=module, param=param, extra_args=self.extra_args
            )
            if not isinstance(outputs, (list, tuple)):
                # Make sure we are unpacking a list whilst post-proc
                outputs = [outputs]

        outputs = self._arguments_postproc(args, outputs, data_keys=self.transform_op.data_keys)  # type: ignore

        if isinstance(original_keys, tuple):
            result = {k: v for v, k in zip(outputs, original_keys)}
            if invalid_data:
                result.update(invalid_data)
            return result

        if len(outputs) == 1 and isinstance(outputs, list):
            return outputs[0]

        return outputs

    def _validate_args_datakeys(self, *args: DataType, data_keys: List[DataKey]) -> None:
        if len(args) != len(data_keys):
            raise AssertionError(
                f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."
            )
        # TODO: validate args batching, and its consistency

    def _arguments_preproc(self, *args: DataType, data_keys: List[DataKey]) -> List[DataType]:
        inp: List[DataType] = []
        for arg, dcate in zip(args, data_keys):
            if DataKey.get(dcate) in _IMG_OPTIONS:
                arg = cast(Tensor, arg)
                self.input_dtype = arg.dtype
                inp.append(arg)
            elif DataKey.get(dcate) in _MSK_OPTIONS:
                if isinstance(inp, list):
                    arg = cast(List[Tensor], arg)
                    self.mask_dtype = arg[0].dtype
                else:
                    arg = cast(Tensor, arg)
                    self.mask_dtype = arg.dtype
                inp.append(self._preproc_mask(arg))
            elif DataKey.get(dcate) in _KEYPOINTS_OPTIONS:
                inp.append(self._preproc_keypoints(arg, dcate))
            elif DataKey.get(dcate) in _BOXES_OPTIONS:
                inp.append(self._preproc_boxes(arg, dcate))
            elif DataKey.get(dcate) in _CLS_OPTIONS:
                inp.append(arg)
            else:
                raise NotImplementedError(f"input type of {dcate} is not implemented.")
        return inp

    def _arguments_postproc(
        self, in_args: List[DataType], out_args: List[DataType], data_keys: List[DataKey]
    ) -> List[DataType]:
        out: List[DataType] = []
        for in_arg, out_arg, dcate in zip(in_args, out_args, data_keys):
            if DataKey.get(dcate) in _IMG_OPTIONS:
                # It is tensor type already.
                out.append(out_arg)
                # TODO: may add the float to integer (for masks), etc.
            elif DataKey.get(dcate) in _MSK_OPTIONS:
                _out_m = self._postproc_mask(cast(MaskDataType, out_arg))
                out.append(_out_m)

            elif DataKey.get(dcate) in _KEYPOINTS_OPTIONS:
                _out_k = self._postproc_keypoint(in_arg, cast(Keypoints, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, (Tensor, Keypoints)):
                    if isinstance(_out_k, list):
                        _out_k = [i.type(in_arg.dtype) for i in _out_k]
                    else:
                        _out_k = _out_k.type(in_arg.dtype)
                out.append(_out_k)

            elif DataKey.get(dcate) in _BOXES_OPTIONS:
                _out_b = self._postproc_boxes(in_arg, cast(Boxes, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, (Tensor, Boxes)):
                    if isinstance(_out_b, list):
                        _out_b = [i.type(in_arg.dtype) for i in _out_b]
                    else:
                        _out_b = _out_b.type(in_arg.dtype)
                out.append(_out_b)

            elif DataKey.get(dcate) in _CLS_OPTIONS:
                out.append(out_arg)

            else:
                raise NotImplementedError(f"input type of {dcate} is not implemented.")

        return out

    def forward(  # type: ignore[override]
        self,
        *args: Union[DataType, Dict[str, DataType]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, List[DataType], Dict[str, DataType]]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        self.clear_state()

        # Unpack/handle dictionary args
        original_keys = None
        if len(args) == 1 and isinstance(args[0], dict):
            original_keys, data_keys, args, invalid_data = self._preproc_dict_data(args[0])

        self.transform_op.data_keys = self.transform_op.preproc_datakeys(data_keys)

        self._validate_args_datakeys(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        in_args = self._arguments_preproc(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        if params is None:
            # image data must exist if params is not provided.
            if DataKey.INPUT in self.transform_op.data_keys:
                inp = in_args[self.transform_op.data_keys.index(DataKey.INPUT)]
                if not isinstance(inp, (Tensor,)):
                    raise ValueError(f"`INPUT` should be a tensor but `{type(inp)}` received.")
                # A video input shall be BCDHW while an image input shall be BCHW
                if self.contains_video_sequential or self.contains_3d_augmentation:
                    _, out_shape = self.autofill_dim(inp, dim_range=(3, 5))
                else:
                    _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
                params = self.forward_parameters(out_shape)
            else:
                raise ValueError("`params` must be provided whilst INPUT is not in data_keys.")

        outputs: Union[Tensor, List[DataType]] = in_args
        for param in params:
            module = self.get_submodule(param.name)
            outputs = self.transform_op.transform(  # type: ignore
                *outputs, module=module, param=param, extra_args=self.extra_args
            )
            if not isinstance(outputs, (list, tuple)):
                # Make sure we are unpacking a list whilst post-proc
                outputs = [outputs]
            self._update_transform_matrix_by_module(module)

        outputs = self._arguments_postproc(args, outputs, data_keys=self.transform_op.data_keys)  # type: ignore
        # Restore it back
        self.transform_op.data_keys = self.data_keys

        self._params = params

        if isinstance(original_keys, tuple):
            result = {k: v for v, k in zip(outputs, original_keys)}
            if invalid_data:
                result.update(invalid_data)
            return result

        if len(outputs) == 1 and isinstance(outputs, list):
            return outputs[0]

        return outputs

    def __call__(
        self,
        *inputs: Any,
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: str = "tensor",
        **kwargs: Any,
    ) -> Any:
        """Overwrites the __call__ function to handle various inputs.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """

        # Wrap the forward method with the decorator
        if not self._disable_features:
            # TODO: Some more behaviour for AugmentationSequential needs to be revisited later
            # e.g. We convert only images, etc.
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super(ImageSequential, self).__call__)
            _output_image = decorated_forward(*inputs, **kwargs)

            if len(inputs) == 1 and isinstance(inputs[0], dict):
                original_keys, in_data_keys, inputs, invalid_data = self._preproc_dict_data(inputs[0])
            else:
                in_data_keys = kwargs.get("data_keys", self.data_keys)
            data_keys = self.transform_op.preproc_datakeys(in_data_keys)

            if len(data_keys) > 1 and DataKey.INPUT in data_keys:
                # NOTE: we may update it later for more supports of drawing boxes, etc.
                idx = data_keys.index(DataKey.INPUT)
                if output_type == "tensor":
                    self._output_image = _output_image
                    if isinstance(_output_image, dict):
                        self._output_image[original_keys[idx]] = self._detach_tensor_to_cpu(
                            _output_image[original_keys[idx]]
                        )
                    else:
                        self._output_image[idx] = self._detach_tensor_to_cpu(_output_image[idx])
                elif isinstance(_output_image, dict):
                    self._output_image[original_keys[idx]] = _output_image[original_keys[idx]]
                else:
                    self._output_image[idx] = _output_image[idx]
            else:
                self._output_image = _output_image
        else:
            _output_image = super(ImageSequential, self).__call__(*inputs, **kwargs)
        return _output_image

    def _preproc_dict_data(
        self, data: Dict[str, DataType]
    ) -> Tuple[Tuple[str, ...], List[DataKey], Tuple[DataType, ...], Optional[Dict[str, Any]]]:
        if self.data_keys is not None:
            raise ValueError("If you are using a dictionary as input, the data_keys should be None.")

        keys = tuple(data.keys())
        data_keys, invalid_keys = self._read_datakeys_from_dict(keys)
        invalid_data = {i: data.pop(i) for i in invalid_keys} if invalid_keys else None
        keys = tuple(k for k in keys if k not in invalid_keys) if invalid_keys else keys
        data_unpacked = tuple(data.values())

        return keys, data_keys, data_unpacked, invalid_data

    def _read_datakeys_from_dict(self, keys: Sequence[str]) -> Tuple[List[DataKey], Optional[List[str]]]:
        def retrieve_key(key: str) -> DataKey:
            """Try to retrieve the datakey value by matching `<datakey>*`"""
            # Alias cases, like INPUT, will not be get by the enum iterator.
            if key.upper().startswith("INPUT"):
                return DataKey.INPUT

            for dk in DataKey:
                if key.upper() in {"BBOX_XYXY", "BBOX_XYWH"}:
                    return DataKey.get(key.upper())
                if key.upper().startswith(dk.name):
                    return DataKey.get(dk.name)

            allowed_dk = " | ".join(f"`{d.name}`" for d in DataKey)
            raise ValueError(
                f"Your input data dictionary keys should start with some of datakey values: {allowed_dk}. Got `{key}`"
            )

        valid_data_keys = []
        invalid_keys = []
        for k in keys:
            try:
                valid_data_keys.append(DataKey.get(retrieve_key(k)))
            except ValueError:
                invalid_keys.append(k)

        return valid_data_keys, invalid_keys

    def _preproc_mask(self, arg: MaskDataType) -> MaskDataType:
        if isinstance(arg, list):
            new_arg = []
            for a in arg:
                a_new = a.to(self.input_dtype) if self.input_dtype else a.to(torch.float)
                new_arg.append(a_new)
            return new_arg

        else:
            arg = arg.to(self.input_dtype) if self.input_dtype else arg.to(torch.float)
        return arg

    def _postproc_mask(self, arg: MaskDataType) -> MaskDataType:
        if isinstance(arg, list):
            new_arg = []
            for a in arg:
                a_new = a.to(self.mask_dtype) if self.mask_dtype else a.to(torch.float)
                new_arg.append(a_new)
            return new_arg

        else:
            arg = arg.to(self.mask_dtype) if self.mask_dtype else arg.to(torch.float)
        return arg

    def _preproc_boxes(self, arg: DataType, dcate: DataKey) -> Boxes:
        if DataKey.get(dcate) in [DataKey.BBOX]:
            mode = "vertices_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYXY]:
            mode = "xyxy_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYWH]:
            mode = "xywh"
        else:
            raise ValueError(f"Unsupported mode `{DataKey.get(dcate).name}`.")
        if isinstance(arg, (Boxes,)):
            return arg
        elif self.contains_video_sequential:
            arg = cast(Tensor, arg)
            return VideoBoxes.from_tensor(arg)
        elif self.contains_3d_augmentation:
            raise NotImplementedError("3D box handlers are not yet supported.")
        else:
            arg = cast(Tensor, arg)
            return Boxes.from_tensor(arg, mode=mode)

    def _postproc_boxes(self, in_arg: DataType, out_arg: Boxes, dcate: DataKey) -> Union[Tensor, List[Tensor], Boxes]:
        if DataKey.get(dcate) in [DataKey.BBOX]:
            mode = "vertices_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYXY]:
            mode = "xyxy_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYWH]:
            mode = "xywh"
        else:
            raise ValueError(f"Unsupported mode `{DataKey.get(dcate).name}`.")

        # TODO: handle 3d scenarios
        if isinstance(in_arg, (Boxes,)):
            return out_arg
        else:
            return out_arg.to_tensor(mode=mode)

    def _preproc_keypoints(self, arg: DataType, dcate: DataKey) -> Keypoints:
        dtype = None

        if self.contains_video_sequential:
            arg = cast(Union[Tensor, List[Tensor]], arg)
            if isinstance(arg, list):
                if not torch.is_floating_point(arg[0]):
                    dtype = arg[0].dtype
                    arg = [a.float() for a in arg]
            elif not torch.is_floating_point(arg):
                dtype = arg.dtype
                arg = arg.float()
            video_result = VideoKeypoints.from_tensor(arg)
            return video_result.type(dtype) if dtype else video_result
        elif self.contains_3d_augmentation:
            raise NotImplementedError("3D keypoint handlers are not yet supported.")
        elif isinstance(arg, (Keypoints,)):
            return arg
        else:
            arg = cast(Tensor, arg)
            if not torch.is_floating_point(arg):
                dtype = arg.dtype
                arg = arg.float()
            # TODO: Add List[Tensor] in the future.
            result = Keypoints.from_tensor(arg)
            return result.type(dtype) if dtype else result

    def _postproc_keypoint(
        self, in_arg: DataType, out_arg: Keypoints, dcate: DataKey
    ) -> Union[Tensor, List[Tensor], Keypoints]:
        if isinstance(in_arg, (Keypoints,)):
            return out_arg
        else:
            return out_arg.to_tensor()
