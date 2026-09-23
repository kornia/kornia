# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn

from kornia.augmentation._2d.base import RigidAffineAugmentationBase2D
from kornia.augmentation._3d.base import AugmentationBase3D, RigidAffineAugmentationBase3D
from kornia.augmentation.base import _AugmentationBase
from kornia.constants import DataKey, Resample
from kornia.core.ops import eye_like
from kornia.core.utils import is_autocast_enabled, is_exporting
from kornia.geometry.boxes import Boxes, VideoBoxes
from kornia.geometry.keypoints import Keypoints, VideoKeypoints

from .base import TransformMatrixMinIn
from .image import ImageSequential
from .ops import AugmentationSequentialOps, DataType
from .params import ParamItem
from .patch import PatchSequential
from .video import VideoSequential

__all__ = ["AugmentationSequential"]

# Dynamo cannot trace membership checks against a Python set; tuples keep these constants
# capture-friendly while preserving the membership-only behavior used below.
_BOXES_OPTIONS = (DataKey.BBOX, DataKey.BBOX_XYXY, DataKey.BBOX_XYWH)
_KEYPOINTS_OPTIONS = (DataKey.KEYPOINTS,)
_IMG_OPTIONS = (DataKey.INPUT, DataKey.IMAGE)
_MSK_OPTIONS = (DataKey.MASK,)
_CLS_OPTIONS = (DataKey.CLASS, DataKey.LABEL)

MaskDataType = Union[torch.Tensor, List[torch.Tensor]]


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

        random_apply: randomly select children to apply. The selected children run in random order and may repeat
                      when the requested count exceeds the sum of the selection weights.
                      If int, a fixed number of transformations will be selected.
                      If (a,), x number of transformations (a <= x <= len(args)) will be selected.
                      If (a, b), x number of transformations (a <= x <= b) will be selected.
                      If True, the whole list of args will be processed as a sequence in a random order.
                      If False, the whole list of args will be processed as a sequence in original order.

        transformation_matrix_mode: computation mode for the chained transformation matrix, via `.transform_matrix`
                                    attribute.
                                    If `silent`, the default, the transformation matrix is computed silently and a
                                    direct non-rigid module is skipped rather than treated as an identity. Nested
                                    containers are not handled consistently; see the Convention note below.
                                    If `rigid`, transformation matrix will be computed silently and the non-rigid
                                    modules will trigger errors.
                                    If `skip`, transformation matrix will be totally ignored.
                                    The validator also accepts `silence`, an alias that behaves
                                    like `silent`; any other value raises ``ValueError``.

        extra_args: a dict keyed by ``kornia.constants.DataKey`` that **replaces** the default
                    ``{DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}}`` rather than
                    merging into it, so an override that omits a key drops that key's default. An empty dict
                    is falsy and restores the default. ``DataKey.IMAGE`` honours both entries and
                    ``DataKey.KEYPOINTS`` honours neither. For ``DataKey.MASK`` on a 2D geometric augmentation
                    that uses the base mask path the ``resample`` entry is honoured in both directions, and a
                    dict without one resamples masks with nearest neighbour. ``align_corners`` is
                    handler-dependent: some warps honor it, but resize mask paths replace it.
                    With :class:`~kornia.augmentation.RandomResizedCrop`,
                    boolean ``align_corners`` overrides raise ``ValueError`` in the default ``cropping_mode='slice'``
                    mask path; ``cropping_mode='resample'`` accepts them. ``None`` works in both modes.
                    :class:`~kornia.augmentation.RandomElasticTransform` has its own mask path and honours
                    both entries, but requires a ``kornia.constants.Resample`` member rather than a
                    string.

    Convention:
        - each child keeps the contract of its own base and class; mix and 3D children do not inherit every
          :class:`~kornia.augmentation.AugmentationBase2D` convention.
        - ``data_keys`` names one entry per positional argument, case-insensitively: ``input`` (alias
          ``image``), ``mask``, ``bbox``, ``bbox_xyxy``, ``bbox_xywh``, ``keypoints`` and ``label`` (alias
          ``class``); any other name raises ``KeyError``. With ``data_keys=None`` the call takes a dict whose
          keys are these names, optionally followed by a ``_`` or ``-`` suffix (``mask_2``, ``class_id``). The
          longest matching name wins and a suffix needs its separator: ``bbox_xyxy2`` is a ``bbox`` key and must
          hold vertices. Unrecognized keys (``images``, ``masks``, ``labels``) are returned unchanged, without a
          warning. The input dict is not modified.
        - the layouts are ``(B, C, H, W)`` for images and masks, ``(B, N, 4, 2)`` vertices for ``bbox``,
          ``(B, N, 4)`` for ``bbox_xyxy`` and ``bbox_xywh``, and ``(B, N, 2)`` in ``(x, y)`` for ``keypoints``;
          ``N = 0`` is accepted. 3D inputs are ``(D, H, W)`` or ``(B, C, D, H, W)``; rank 4 is rejected as
          ambiguous. A ``(B, H, W)`` mask is returned as ``(B, 1, H, W)``. A wrong input rank raises
          ``RuntimeError`` here rather than the ``ValueError`` of a bare augmentation
          (`#4424 <https://github.com/kornia/kornia/issues/4424>`_).
        - boxes use the inclusive ``xyxy_plus`` convention of :class:`~kornia.geometry.boxes.Boxes`. Flips map
          ``x' = W - 1 - x`` and ``y' = H - 1 - y`` for every key, as :func:`~kornia.geometry.transform.hflip`
          does. Labels pass through geometric steps untouched.
        - masks are resampled with nearest interpolation (padding can still add the fill value) in the image's
          working dtype and come back in their own dtype; integer labels that the working dtype cannot represent
          are rounded (`#4478 <https://github.com/kornia/kornia/issues/4478>`_).
        - a ``mask`` argument can be a list of tensors with different channel counts, but its batch handling
          has limitations. Each list entry uses only ``batch_prob[i]`` as its gate, including for intensity
          children. Per-sample list tensors are unsupported by warp operations, and full-batch tensors in that
          list can become desynchronized from the image when the gate differs across samples. A list longer
          than the batch raises ``IndexError``. Use separate ``mask`` data keys for separate full-batch masks.
          Tracked in `#4477 <https://github.com/kornia/kornia/issues/4477>`_.
        - supported geometric data-key handlers share the recorded transform, subject to the mask limitations
          above. Custom rigid subclasses are not dispatched solely because they supply a matrix
          (`#4481 <https://github.com/kornia/kornia/issues/4481>`_). A non-rigid child has no matrix, so the
          coordinate keys are left unchanged; see the warning below.
        - ``.inverse()`` undoes the 2D geometric steps and leaves intensity and non-rigid steps applied. Slice-mode
          crops and 3D geometric children raise ``NotImplementedError``. Tensor boxes come back as axis-aligned
          enclosures; pass :class:`~kornia.geometry.boxes.Boxes` to keep rotated corners. Content lost to
          cropping, padding or interpolation is not recovered.
        - ``same_on_batch`` and ``keepdim`` default to ``None``, which keeps each child's own setting;
          ``True`` or ``False`` overrides it.
        - ``.transform_matrix`` of a chain holding a nested container is unreliable: it can raise, omit the
          nested transform or return a stale one (`#4476 <https://github.com/kornia/kornia/issues/4476>`_).

    .. warning::
        A non-rigid child silently desynchronizes the coordinate data keys:
        :class:`~kornia.augmentation.RandomElasticTransform` warps the image **and** a ``mask`` key with it,
        but returns keypoints and boxes unchanged, and
        :class:`~kornia.augmentation.RandomThinPlateSpline` and :class:`~kornia.augmentation.RandomFisheye`
        return keypoints and boxes unchanged and raise a bare ``NotImplementedError`` on a ``mask`` key.
        Tracked in `#4420 <https://github.com/kornia/kornia/issues/4420>`_.

    .. note::
        A mix child (e.g. RandomMixUpV2, RandomCutMixV2, RandomMosaic) receives ``mask``, box and ``keypoints``
        keys with the image's parameters; keys it does not implement raise ``NotImplementedError``, as in a
        direct call. A ``class``/``label`` key raises ``NotImplementedError`` from the container.

    .. note::
        See a working example `here <https://www.kornia.org/tutorials/nbs/data_augmentation_sequential.html>`__.

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

    Use separate full-batch mask arguments when their channel counts differ:

        >>> import kornia.augmentation as K
        >>> input = torch.randn(2, 3, 32, 32)
        >>> mask_a = torch.ones(2, 3, 32, 32)
        >>> mask_b = torch.ones(2, 2, 32, 32)
        >>> aug_list = K.AugmentationSequential(
        ...     K.RandomHorizontalFlip(p=1.0),
        ...     data_keys=["input", "mask", "mask"],
        ... )
        >>> out = aug_list(input, mask_a, mask_b)
        >>> [value.shape for value in out]
        [torch.Size([2, 3, 32, 32]), torch.Size([2, 3, 32, 32]), torch.Size([2, 2, 32, 32])]

    With ``data_keys=None``, dictionary keys match data-key names case-insensitively, optionally followed
    by an underscore or hyphen suffix (for example, ``image_2`` or ``bbox_xyxy-left``). The longest matching
    name wins, so coordinate boxes with an underscore/hyphen suffix retain their coordinate format.
    Without that separator, ``bbox_xyxy2`` matches ``bbox`` and requires vertex boxes. ``input`` and ``class`` are
    aliases of ``image`` and ``label``. Unrecognized items are returned without augmentation, and the
    caller's dictionary is left intact.

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
        data_keys: Optional[Union[Sequence[str], Sequence[int], Sequence[DataKey]]] = (DataKey.INPUT,),
        same_on_batch: Optional[bool] = None,
        keepdim: Optional[bool] = None,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
        transformation_matrix_mode: str = "silent",
        extra_args: Optional[Dict[DataKey, Dict[str, Any]]] = None,
    ) -> None:
        self._transform_matrix: Optional[torch.Tensor]
        self._transform_matrices: List[Optional[torch.Tensor]] = []

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
                warnings.warn(
                    "Geometric transformation detected in PatchSeqeuntial, which would break bbox, mask.", stacklevel=1
                )
            if isinstance(arg, VideoSequential):
                self.contains_video_sequential = True
            # NOTE: only for images are supported for 3D.
            if isinstance(arg, AugmentationBase3D):
                self.contains_3d_augmentation = True
        self._transform_matrix = None
        self.extra_args = extra_args or {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}}

    def clear_state(self) -> None:
        """Reset cached params and transformation-matrix state."""
        self._reset_transform_matrix_state()
        return super().clear_state()

    def _update_transform_matrix_for_valid_op(self, module: nn.Module) -> None:
        if not is_exporting():
            self._transform_matrices.append(module.transform_matrix)

    def identity_matrix(self, input: torch.Tensor) -> torch.Tensor:
        """Return identity matrix."""
        if self.contains_3d_augmentation:
            return eye_like(4, input)

        return eye_like(3, input)

    def inverse(  # type: ignore[override]
        self,
        *args: Union[DataType, Dict[str, DataType]],
        params: Optional[List[ParamItem]] = None,
        data_keys: Optional[Union[List[str], List[int], List[DataKey]]] = None,
    ) -> Union[DataType, List[DataType], Dict[str, DataType]]:
        """Reverse the transformation applied.

        See the Convention block on :class:`~kornia.augmentation.container.AugmentationSequential`.

        Number of input tensors must align with the number of``data_keys``. If ``data_keys`` is not set, use
        ``self.data_keys`` by default.
        """
        original_keys = None
        if len(args) == 1 and isinstance(args[0], dict):
            original_keys, data_keys, args, invalid_data = self._preproc_dict_data(cast(Dict[str, DataType], args[0]))

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
            if not isinstance(outputs, list | tuple):
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
        # Resolve this call's image dtype before any mask is converted, so a mask that precedes the image in
        # dictionary insertion order uses it too, rather than the previous call's image dtype (or ``float32`` on
        # a fresh container). It is kept in a local rather than read back from ``self.input_dtype``, so the same
        # conversion happens under ``torch.export``, where that attribute is deliberately left untouched. Masks
        # after an image use the most recent image, as before; a call with no image falls back to the attribute.
        working_dtype = self.input_dtype
        for arg, dcate in zip(args, data_keys):
            if DataKey.get(dcate) in _IMG_OPTIONS:
                working_dtype = cast(torch.Tensor, arg).dtype
                break
        inp: List[DataType] = []
        for arg, dcate in zip(args, data_keys):
            if DataKey.get(dcate) in _IMG_OPTIONS:
                arg = cast(torch.Tensor, arg)
                working_dtype = arg.dtype
                if not is_exporting():
                    self.input_dtype = arg.dtype
                inp.append(arg)
            elif DataKey.get(dcate) in _MSK_OPTIONS:
                # Output dtypes are read back per argument in ``_arguments_postproc``; ``mask_dtype`` only records
                # the last mask's dtype for callers that read the attribute. The test is on ``arg``: it used to be
                # on the accumulator ``inp``, which is always a list, so a tensor mask was indexed at ``arg[0]``
                # and an empty batch raised ``IndexError``. Like ``input_dtype``, the attribute is not written under
                # ``torch.export``: creating an instance attribute during capture fails the export on torch 2.9.
                if not is_exporting():
                    if isinstance(arg, list):
                        if len(arg) > 0:
                            self.mask_dtype = arg[0].dtype
                    else:
                        self.mask_dtype = cast(torch.Tensor, arg).dtype
                inp.append(self._preproc_mask(arg, working_dtype))
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
                # It is torch.Tensor type already.
                out.append(out_arg)
                # TODO: may add the float to integer (for masks), etc.
            elif DataKey.get(dcate) in _MSK_OPTIONS:
                _out_m = self._postproc_mask(cast(MaskDataType, out_arg), cast(MaskDataType, in_arg))
                out.append(_out_m)

            elif DataKey.get(dcate) in _KEYPOINTS_OPTIONS:
                _out_k = self._postproc_keypoint(in_arg, cast(Keypoints, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, torch.Tensor | Keypoints):
                    if isinstance(_out_k, list):
                        _out_k = [i.type(in_arg.dtype) for i in _out_k]
                    else:
                        _out_k = _out_k.type(in_arg.dtype)
                out.append(_out_k)

            elif DataKey.get(dcate) in _BOXES_OPTIONS:
                _out_b = self._postproc_boxes(in_arg, cast(Boxes, out_arg), dcate)
                if is_autocast_enabled() and isinstance(in_arg, torch.Tensor | Boxes):
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
        """Compute multiple tensors simultaneously according to ``self.data_keys``.

        See the Convention block on :class:`~kornia.augmentation.container.AugmentationSequential`.
        """
        self.clear_state()

        # Strip trailing ``None`` positional args. The legacy torch.onnx.export tracer
        # rebinds keyword-only defaults (``params``/``data_keys``) as positional, so
        # ``forward(x)`` arrives as ``forward(x, None, None)``. Real data inputs are
        # always tensors / Boxes / Keypoints / dicts — never ``None``, so this is safe.
        while args and args[-1] is None:
            args = args[:-1]

        # Unpack/handle dictionary args
        original_keys = None
        if len(args) == 1 and isinstance(args[0], dict):
            original_keys, data_keys, args, invalid_data = self._preproc_dict_data(cast(Dict[str, DataType], args[0]))

        self.transform_op.data_keys = self.transform_op.preproc_datakeys(data_keys)

        self._validate_args_datakeys(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        in_args = self._arguments_preproc(*args, data_keys=self.transform_op.data_keys)  # type: ignore

        if DataKey.INPUT in self.transform_op.data_keys:
            inp = in_args[self.transform_op.data_keys.index(DataKey.INPUT)]
            if not isinstance(inp, torch.Tensor):
                raise ValueError(f"`INPUT` should be a torch.Tensor but `{type(inp)}` received.")
            if self.contains_3d_augmentation and len(inp.shape) == 4:
                raise RuntimeError(
                    f"3D augmentations in AugmentationSequential expect input shape "
                    f"(D, H, W) or (B, C, D, H, W), but got {inp.shape}."
                )

        if params is None:
            # image data must exist if params is not provided.
            if DataKey.INPUT in self.transform_op.data_keys:
                inp = in_args[self.transform_op.data_keys.index(DataKey.INPUT)]
                # A video input shall be BCDHW while an image input shall be BCHW
                if self.contains_video_sequential:
                    _, out_shape = self.autofill_dim(inp, dim_range=(3, 5))
                elif self.contains_3d_augmentation:
                    _, out_shape = self.autofill_dim(inp, dim_range=(3, 5))
                else:
                    _, out_shape = self.autofill_dim(inp, dim_range=(2, 4))
                params = self.forward_parameters(out_shape)
            else:
                raise ValueError("`params` must be provided whilst INPUT is not in data_keys.")

        outputs: Union[torch.Tensor, List[DataType]] = in_args
        for param in params:
            module = self.get_submodule(param.name)
            outputs = self.transform_op.transform(  # type: ignore
                *outputs, module=module, param=param, extra_args=self.extra_args
            )
            if not isinstance(outputs, list | tuple):
                # Make sure we are unpacking a list whilst post-proc
                outputs = [outputs]
            self._update_transform_matrix_by_module(module)

        outputs = self._arguments_postproc(args, outputs, data_keys=self.transform_op.data_keys)  # type: ignore
        # Restore it back
        self.transform_op.data_keys = self.data_keys

        if not is_exporting():
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
        output_type: Literal["pt", "numpy", "pil"] = "pt",
        **kwargs: Any,
    ) -> Any:
        """Overwrite the __call__ function to handle various inputs.

        Args:
            inputs: Inputs to operate on.
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('pt', 'numpy', or 'pil').
            kwargs: Additional arguments.

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

            in_data_keys: Optional[List[DataKey]]
            if len(inputs) == 1 and isinstance(inputs[0], dict):
                original_keys, in_data_keys, inputs, _invalid_data = self._preproc_dict_data(inputs[0])
            else:
                in_data_keys = kwargs.get("data_keys", self.data_keys)
            data_keys = self.transform_op.preproc_datakeys(in_data_keys)

            if not is_exporting():
                if len(data_keys) > 1 and DataKey.INPUT in data_keys:
                    idx = data_keys.index(DataKey.INPUT)
                    if output_type == "pt":
                        # ``self._output_image`` already holds ``_output_image`` here, so the old
                        # per-key rebind was a no-op; just store the whole output.
                        self._output_image = _output_image
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
        invalid_data = {i: data[i] for i in invalid_keys} if invalid_keys else None
        keys = tuple(k for k in keys if k not in invalid_keys) if invalid_keys else keys
        data_unpacked = tuple(data[k] for k in keys)

        return keys, data_keys, data_unpacked, invalid_data

    def _read_datakeys_from_dict(self, keys: Sequence[str]) -> Tuple[List[DataKey], Optional[List[str]]]:
        # Include aliases and prefer coordinate box names over their BBOX prefix.
        names = sorted(DataKey.__members__, key=len, reverse=True)

        def retrieve_key(key: str) -> DataKey:
            """Match a data-key name exactly or before an underscore/hyphen suffix."""
            upper_key = key.upper()
            for name in names:
                if upper_key == name or upper_key.startswith((name + "_", name + "-")):
                    return DataKey.get(name)
            raise ValueError(f"Unrecognized data dictionary key: {key}")

        valid_data_keys = []
        invalid_keys = []
        for k in keys:
            try:
                valid_data_keys.append(DataKey.get(retrieve_key(k)))
            except ValueError:
                invalid_keys.append(k)

        return valid_data_keys, invalid_keys

    def _preproc_mask(self, arg: MaskDataType, dtype: Optional[torch.dtype]) -> MaskDataType:
        # ``dtype`` is the calling image's working dtype, resolved by ``_arguments_preproc``; ``float32`` when the
        # call has no image and no earlier call recorded one.
        working = dtype if dtype is not None else torch.float
        if isinstance(arg, list):
            return [a.to(working) for a in arg]
        return arg.to(working)

    def _postproc_mask(self, arg: MaskDataType, like: MaskDataType) -> MaskDataType:
        # Each mask output goes back to the dtype of its own argument, per element for a list. A single shared
        # dtype would cast every mask to whichever mask came last: an integer semantic mask followed by a boolean
        # one came back boolean, and its labels collapsed to ``True``.
        if isinstance(arg, list):
            likes = like if isinstance(like, list) else [like] * len(arg)
            return [a.to(ref.dtype) for a, ref in zip(arg, likes)]
        ref = like[0] if isinstance(like, list) else like
        return arg.to(ref.dtype)

    def _preproc_boxes(self, arg: DataType, dcate: DataKey) -> Boxes:
        if DataKey.get(dcate) in [DataKey.BBOX]:
            mode = "vertices_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYXY]:
            mode = "xyxy_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYWH]:
            mode = "xywh"
        else:
            raise ValueError(f"Unsupported mode `{DataKey.get(dcate).name}`.")
        if isinstance(arg, Boxes):
            return arg
        if self.contains_video_sequential:
            arg = cast(torch.Tensor, arg)
            return VideoBoxes.from_tensor(arg)
        if self.contains_3d_augmentation:
            raise NotImplementedError("3D box handlers are not yet supported.")
        arg = cast(torch.Tensor, arg)
        return Boxes.from_tensor(arg, mode=mode)

    def _postproc_boxes(
        self, in_arg: DataType, out_arg: Boxes, dcate: DataKey
    ) -> Union[torch.Tensor, List[torch.Tensor], Boxes]:
        if DataKey.get(dcate) in [DataKey.BBOX]:
            mode = "vertices_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYXY]:
            mode = "xyxy_plus"
        elif DataKey.get(dcate) in [DataKey.BBOX_XYWH]:
            mode = "xywh"
        else:
            raise ValueError(f"Unsupported mode `{DataKey.get(dcate).name}`.")

        # TODO: handle 3d scenarios
        if isinstance(in_arg, Boxes):
            return out_arg

        return out_arg.to_tensor(mode=mode)

    def _preproc_keypoints(self, arg: DataType, dcate: DataKey) -> Keypoints:
        dtype = None

        if self.contains_video_sequential:
            arg = cast(Union[torch.Tensor, List[torch.Tensor]], arg)
            if isinstance(arg, list):
                if not torch.is_floating_point(arg[0]):
                    dtype = arg[0].dtype
                    arg = [a.float() for a in arg]
            elif not torch.is_floating_point(arg):
                dtype = arg.dtype
                arg = arg.float()
            video_result = VideoKeypoints.from_tensor(arg)
            return video_result.type(dtype) if dtype else video_result
        if self.contains_3d_augmentation:
            raise NotImplementedError("3D keypoint handlers are not yet supported.")
        if isinstance(arg, Keypoints):
            return arg
        arg = cast(torch.Tensor, arg)
        if not torch.is_floating_point(arg):
            dtype = arg.dtype
            arg = arg.float()
        # TODO: Add List[torch.Tensor] in the future.
        result = Keypoints.from_tensor(arg)
        return result.type(dtype) if dtype else result

    def _postproc_keypoint(
        self, in_arg: DataType, out_arg: Keypoints, dcate: DataKey
    ) -> Union[torch.Tensor, List[torch.Tensor], Keypoints]:
        if isinstance(in_arg, Keypoints):
            return out_arg

        return out_arg.to_tensor()
