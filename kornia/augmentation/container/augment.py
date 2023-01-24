import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from kornia.augmentation import AugmentationBase3D, RigidAffineAugmentationBase2D, RigidAffineAugmentationBase3D
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.image import ImageSequential, ParamItem
from kornia.augmentation.container.ops import AugmentationSequentialOps, DataType
from kornia.augmentation.container.patch import PatchSequential
from kornia.augmentation.container.video import VideoSequential
from kornia.constants import DataKey, Resample
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes, VideoBoxes
from kornia.geometry.keypoints import Keypoints, VideoKeypoints
from kornia.utils import eye_like, is_autocast_enabled

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
        transformation_matrix: computation mode for the chained transformation matrix,
            via `.transform_matrix` attribute.
            If `silence`, transformation matrix will be computed silently and the non-rigid modules
                will be ignored as identity transformations.
            If `rigid`, transformation matrix will be computed silently and the non-rigid modules
                will trigger errors.
            If `skip`, transformation matrix will be totally ignored.
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
    """

    _transform_matrix: Optional[Tensor]
    _transform_matrices: List[Tensor] = []

    _boxes_options = {DataKey.BBOX, DataKey.BBOX_XYXY, DataKey.BBOX_XYWH}
    _keypoints_options = {DataKey.KEYPOINTS}
    _img_msk_options = {DataKey.INPUT, DataKey.MASK}

    def __init__(
        self,
        *args: Union[_AugmentationBase, ImageSequential],
        data_keys: List[Union[str, int, DataKey]] = [DataKey.INPUT],
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        transformation_matrix: str = "silence",
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
            raise AssertionError(f"`data_keys` must be in {DataKey}. Got {self.data_keys}.")

        if self.data_keys[0] != DataKey.INPUT:
            raise NotImplementedError(f"The first input must be {DataKey.INPUT}.")

        self.transform_op = AugmentationSequentialOps(self.data_keys)

        _valid_transformation_matrix_args = ["silence", "rigid", "skip"]
        if transformation_matrix not in _valid_transformation_matrix_args:
            raise ValueError(
                f"`transformation_matrix` has to be one of {_valid_transformation_matrix_args}. "
                f"Got {transformation_matrix}."
            )
        self._transformation_matrix_arg = transformation_matrix

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

    def clear_state(self) -> None:
        self._transform_matrix = None
        self._transform_matrices = []
        return super().clear_state()

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        # In AugmentationSequential, the parent class is accessed first.
        # So that it was None in the beginning. We hereby use lazy computation here.
        if self._transform_matrix is None and len(self._transform_matrices) != 0:
            self._transform_matrix = self._transform_matrices[0]
            for mat in self._transform_matrices[1:]:
                self._update_transform_matrix(mat)
        return self._transform_matrix

    def _update_transform_matrix_by_module(self, module: Module) -> None:
        if self._transformation_matrix_arg == "skip":
            return
        if isinstance(module, (RigidAffineAugmentationBase2D, RigidAffineAugmentationBase3D, AugmentationSequential)):
            # Passed in pointer, allows lazy transformation matrix computation
            self._transform_matrices.append(module.transform_matrix)  # type: ignore
        elif self._transformation_matrix_arg == "rigid":
            raise RuntimeError(
                f"Non-rigid module `{module}` is not supported under `rigid` computation mode. "
                "Please either update the module or change the `transformation_matrix` argument."
            )

    def _update_transform_matrix(self, transform_matrix: Tensor) -> None:
        if self._transform_matrix is None:
            self._transform_matrix = transform_matrix
        else:
            self._transform_matrix = transform_matrix @ self._transform_matrix

    def identity_matrix(self, input: Tensor) -> Tensor:
        """Return identity matrix."""
        if self.contains_3d_augmentation:
            return eye_like(4, input)
        else:
            return eye_like(3, input)

    def inverse(  # type: ignore[override]
        self,
        *args: DataType,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[DataType, List[DataType]]:
        """Reverse the transformation applied.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys`` is not set, use
        ``self.data_keys`` by default.
        """
        self.transform_op.data_keys = self.transform_op.preproc_datakeys(data_keys)

        self._validate_args_datakeys(*args, data_keys=self.transform_op.data_keys)

        in_args = self._arguments_preproc(*args, data_keys=self.transform_op.data_keys)

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

        if len(outputs) == 1 and isinstance(outputs, list):
            return outputs[0]

        return outputs

    def _validate_args_datakeys(self, *args: DataType, data_keys: List[DataKey]):
        if len(args) != len(data_keys):
            raise AssertionError(
                f"The number of inputs must align with the number of data_keys. Got {len(args)} and {len(data_keys)}."
            )
        # TODO: validate args batching, and its consistency

    def _arguments_preproc(self, *args: DataType, data_keys: List[DataKey]) -> List[DataType]:
        inp: List[DataType] = []
        for arg, dcate in zip(args, data_keys):
            if DataKey.get(dcate) in self._img_msk_options:
                inp.append(arg)
            elif DataKey.get(dcate) in self._keypoints_options:
                inp.append(self._preproc_keypoints(arg, dcate))
            elif DataKey.get(dcate) in self._boxes_options:
                inp.append(self._preproc_boxes(arg, dcate))
            else:
                raise NotImplementedError(f"input type of {dcate} is not implemented.")
        return inp

    def _arguments_postproc(
        self, in_args: List[DataType], out_args: List[DataType], data_keys: List[DataKey]
    ) -> List[DataType]:
        out: List[DataType] = []
        for in_arg, out_arg, dcate in zip(in_args, out_args, data_keys):
            if DataKey.get(dcate) in self._img_msk_options:
                # It is tensor type already.
                out.append(out_arg)
                # TODO: may add the float to integer (for masks), etc.

            elif DataKey.get(dcate) in self._keypoints_options:
                _out_k = self._postproc_keypoint(in_arg, cast(Keypoints, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, (Tensor, Keypoints)):
                    if isinstance(_out_k, list):
                        _out_k = [i.type(in_arg.dtype) for i in _out_k]
                    else:
                        _out_k = _out_k.type(in_arg.dtype)
                out.append(_out_k)

            elif DataKey.get(dcate) in self._boxes_options:
                _out_b = self._postproc_boxes(in_arg, cast(Boxes, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, (Tensor, Boxes)):
                    if isinstance(_out_b, list):
                        _out_b = [i.type(in_arg.dtype) for i in _out_b]
                    else:
                        _out_b = _out_b.type(in_arg.dtype)
                out.append(_out_b)

            else:
                raise NotImplementedError(f"input type of {dcate} is not implemented.")

        return out

    def forward(  # type: ignore[override]
        self,
        *args: DataType,
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[List[Union[str, int, DataKey]]] = None,
    ) -> Union[DataType, List[DataType]]:
        """Compute multiple tensors simultaneously according to ``self.data_keys``."""
        self.clear_state()

        self.transform_op.data_keys = self.transform_op.preproc_datakeys(data_keys)

        self._validate_args_datakeys(*args, data_keys=self.transform_op.data_keys)

        in_args = self._arguments_preproc(*args, data_keys=self.transform_op.data_keys)

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

        if len(outputs) == 1 and isinstance(outputs, list):
            return outputs[0]

        return outputs

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
        if self.contains_video_sequential:
            arg = cast(Union[Tensor, List[Tensor]], arg)
            return VideoKeypoints.from_tensor(arg)
        elif self.contains_3d_augmentation:
            raise NotImplementedError("3D keypoint handlers are not yet supported.")
        else:
            arg = cast(Tensor, arg)
            # TODO: Add List[Tensor] in the future.
            return Keypoints.from_tensor(arg)

    def _postproc_keypoint(
        self, in_arg: DataType, out_arg: Keypoints, dcate: DataKey
    ) -> Union[Tensor, List[Tensor], Keypoints]:
        if isinstance(in_arg, (Keypoints,)):
            return out_arg
        else:
            return out_arg.to_tensor()
