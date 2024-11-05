from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

import kornia.augmentation as K
from kornia.augmentation.base import _AugmentationBase
from kornia.augmentation.container.base import SequentialBase
from kornia.augmentation.container.image import ImageSequential, _get_new_batch_shape
from kornia.core import Module, Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .params import ParamItem

__all__ = ["VideoSequential"]


class VideoSequential(ImageSequential):
    r"""VideoSequential for processing 5-dim video data like (B, T, C, H, W) and (B, C, T, H, W).

    `VideoSequential` is used to replace `nn.Sequential` for processing video data augmentations.
    By default, `VideoSequential` enabled `same_on_frame` to make sure the same augmentations happen
    across temporal dimension. Meanwhile, it will not affect other augmentation behaviours like the
    settings on `same_on_batch`, etc.

    Args:
        *args: a list of augmentation module.
        data_format: only BCTHW and BTCHW are supported.
        same_on_frame: apply the same transformation across the channel per frame.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If None, the whole list of args will be processed as a sequence.

    Note:
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.

    Example:
        If set `same_on_frame` to True, we would expect the same augmentation has been applied to each
        timeframe.

        >>> import kornia
        >>> input = torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1)
        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     random_apply=10,
        ...     data_format="BCTHW",
        ...     same_on_frame=True)
        >>> output = aug_list(input)
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(True)
        >>> (output[0, :, 1] == output[0, :, 2]).all()
        tensor(True)
        >>> (output[0, :, 2] == output[0, :, 3]).all()
        tensor(True)

        If set `same_on_frame` to False:

        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.augmentation.RandomMixUpV2(p=1.0),
        ... data_format="BCTHW",
        ... same_on_frame=False)
        >>> output = aug_list(input)
        >>> output.shape
        torch.Size([2, 3, 4, 5, 6])
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(False)

        Reproduce with provided params.
        >>> out2 = aug_list(input, params=aug_list._params)
        >>> torch.equal(output, out2)
        True

    Perform ``OneOf`` transformation with ``random_apply=1`` and ``random_apply_weights`` in ``VideoSequential``.

        >>> import kornia
        >>> input, label = torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1), torch.tensor([0, 1])
        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.augmentation.RandomMixUpV2(p=1.0),
        ... data_format="BCTHW",
        ... same_on_frame=False,
        ... random_apply=1,
        ... random_apply_weights=[0.5, 0.3, 0.8]
        ... )
        >>> out = aug_list(input)
        >>> out.shape
        torch.Size([2, 3, 4, 5, 6])
    """

    # TODO: implement transform_matrix

    def __init__(
        self,
        *args: Module,
        data_format: str = "BTCHW",
        same_on_frame: bool = True,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
        random_apply_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            *args,
            same_on_batch=None,
            keepdim=None,
            random_apply=random_apply,
            random_apply_weights=random_apply_weights,
        )
        self.same_on_frame = same_on_frame
        self.data_format = data_format.upper()
        if self.data_format not in ["BCTHW", "BTCHW"]:
            raise AssertionError(f"Only `BCTHW` and `BTCHW` are supported. Got `{data_format}`.")
        self._temporal_channel: int
        if self.data_format == "BCTHW":
            self._temporal_channel = 2
        elif self.data_format == "BTCHW":
            self._temporal_channel = 1

    def __infer_channel_exclusive_batch_shape__(self, batch_shape: torch.Size, chennel_index: int) -> torch.Size:
        # Fix mypy complains: error: Incompatible return value type (got "Tuple[int, ...]", expected "Size")
        return cast(torch.Size, batch_shape[:chennel_index] + batch_shape[chennel_index + 1 :])

    def __repeat_param_across_channels__(self, param: Tensor, frame_num: int) -> Tensor:
        """Repeat parameters across channels.

        The input is shaped as (B, ...), while to output (B * same_on_frame, ...), which
        to guarantee that the same transformation would happen for each frame.

        (B1, B2, ..., Bn) => (B1, ... B1, B2, ..., B2, ..., Bn, ..., Bn)
                              | ch_size | | ch_size |  ..., | ch_size |
        """
        repeated = param[:, None, ...].repeat(1, frame_num, *([1] * len(param.shape[1:])))
        return repeated.reshape(-1, *list(param.shape[1:]))

    def _input_shape_convert_in(self, input: Tensor, frame_num: int) -> Tensor:
        # Convert any shape to (B, T, C, H, W)
        if self.data_format == "BCTHW":
            # Convert (B, C, T, H, W) to (B, T, C, H, W)
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass

        input = input.reshape(-1, *input.shape[2:])
        return input

    def _input_shape_convert_back(self, input: Tensor, frame_num: int) -> Tensor:
        input = input.view(-1, frame_num, *input.shape[1:])
        if self.data_format == "BCTHW":
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass

        return input

    def forward_parameters(self, batch_shape: torch.Size) -> List[ParamItem]:
        frame_num = batch_shape[self._temporal_channel]
        named_modules = self.get_forward_sequence()
        # Got param generation shape to (B, C, H, W). Ignoring T.
        batch_shape = self.__infer_channel_exclusive_batch_shape__(batch_shape, self._temporal_channel)

        if not self.same_on_frame:
            # Overwrite param generation shape to (B * T, C, H, W).
            batch_shape = torch.Size([batch_shape[0] * frame_num, *batch_shape[1:]])

        params = []
        for name, module in named_modules:
            if isinstance(module, K.RandomCrop):
                mod_param = module.forward_parameters(batch_shape)
                if self.same_on_frame:
                    mod_param["src"] = mod_param["src"].repeat(frame_num, 1, 1)
                    mod_param["dst"] = mod_param["dst"].repeat(frame_num, 1, 1)
                param = ParamItem(name, mod_param)
            elif isinstance(module, (SequentialBase,)):
                seq_param = module.forward_parameters(batch_shape)
                if self.same_on_frame:
                    raise ValueError("Sequential is currently unsupported for ``same_on_frame``.")
                param = ParamItem(name, seq_param)
            elif isinstance(module, (_AugmentationBase, K.MixAugmentationBaseV2)):
                mod_param = module.forward_parameters(batch_shape)
                if self.same_on_frame:
                    for k, v in mod_param.items():
                        # TODO: revise ColorJiggle and ColorJitter order param in the future to align the standard.
                        if k == "order" and (isinstance(module, (K.ColorJiggle, K.ColorJitter))):
                            continue
                        if k == "forward_input_shape":
                            mod_param.update({k: v})
                            continue
                        mod_param.update({k: self.__repeat_param_across_channels__(v, frame_num)})
                param = ParamItem(name, mod_param)
            else:
                param = ParamItem(name, None)
            batch_shape = _get_new_batch_shape(param, batch_shape)
            params.append(param)
        return params

    def transform_inputs(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        frame_num: int = input.size(self._temporal_channel)
        input = self._input_shape_convert_in(input, frame_num)

        input = super().transform_inputs(input, params, extra_args=extra_args)

        input = self._input_shape_convert_back(input, frame_num)
        return input

    def inverse_inputs(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        frame_num: int = input.size(self._temporal_channel)
        input = self._input_shape_convert_in(input, frame_num)

        input = super().inverse_inputs(input, params, extra_args=extra_args)

        input = self._input_shape_convert_back(input, frame_num)
        return input

    def transform_masks(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        frame_num: int = input.size(self._temporal_channel)
        input = self._input_shape_convert_in(input, frame_num)

        input = super().transform_masks(input, params, extra_args=extra_args)

        input = self._input_shape_convert_back(input, frame_num)
        return input

    def inverse_masks(self, input: Tensor, params: List[ParamItem], extra_args: Dict[str, Any] = {}) -> Tensor:
        frame_num: int = input.size(self._temporal_channel)
        input = self._input_shape_convert_in(input, frame_num)

        input = super().inverse_masks(input, params, extra_args=extra_args)

        input = self._input_shape_convert_back(input, frame_num)
        return input

    def transform_boxes(  # type: ignore[override]
        self, input: Union[Tensor, Boxes], params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Union[Tensor, Boxes]:
        """Transform bounding boxes.

        Args:
            input: tensor with shape :math:`(B, T, N, 4, 2)`.
                If input is a `Keypoints` type, the internal shape is :math:`(B * T, N, 4, 2)`.
        """
        if isinstance(input, Tensor):
            batchsize, frame_num = input.size(0), input.size(1)
            input = Boxes.from_tensor(input.view(-1, input.size(2), input.size(3), input.size(4)), mode="vertices_plus")
            input = super().transform_boxes(input, params, extra_args=extra_args)
            input = input.data.view(batchsize, frame_num, -1, 4, 2)
        else:
            input = super().transform_boxes(input, params, extra_args=extra_args)
        return input

    def inverse_boxes(  # type: ignore[override]
        self, input: Union[Tensor, Boxes], params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Union[Tensor, Boxes]:
        """Transform bounding boxes.

        Args:
            input: tensor with shape :math:`(B, T, N, 4, 2)`.
                If input is a `Keypoints` type, the internal shape is :math:`(B * T, N, 4, 2)`.
        """
        if isinstance(input, Tensor):
            batchsize, frame_num = input.size(0), input.size(1)
            input = Boxes.from_tensor(input.view(-1, input.size(2), input.size(3), input.size(4)), mode="vertices_plus")
            input = super().inverse_boxes(input, params, extra_args=extra_args)
            input = input.data.view(batchsize, frame_num, -1, 4, 2)
        else:
            input = super().inverse_boxes(input, params, extra_args=extra_args)
        return input

    def transform_keypoints(  # type: ignore[override]
        self, input: Union[Tensor, Keypoints], params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Union[Tensor, Keypoints]:
        """Transform bounding boxes.

        Args:
            input: tensor with shape :math:`(B, T, N, 2)`.
                If input is a `Keypoints` type, the internal shape is :math:`(B * T, N, 2)`.
        """
        if isinstance(input, Tensor):
            batchsize, frame_num = input.size(0), input.size(1)
            input = Keypoints(input.view(-1, input.size(2), input.size(3)))
            input = super().transform_keypoints(input, params, extra_args=extra_args)
            input = input.data.view(batchsize, frame_num, -1, 2)
        else:
            input = super().transform_keypoints(input, params, extra_args=extra_args)
        return input

    def inverse_keypoints(  # type: ignore[override]
        self, input: Union[Tensor, Keypoints], params: List[ParamItem], extra_args: Dict[str, Any] = {}
    ) -> Union[Tensor, Keypoints]:
        """Transform bounding boxes.

        Args:
            input: tensor with shape :math:`(B, T, N, 2)`.
                If input is a `Keypoints` type, the internal shape is :math:`(B * T, N, 2)`.
        """
        if isinstance(input, Tensor):
            frame_num, batchsize = input.size(0), input.size(1)
            input = Keypoints(input.view(-1, input.size(2), input.size(3)))
            input = super().inverse_keypoints(input, params, extra_args=extra_args)
            input = input.data.view(batchsize, frame_num, -1, 2)
        else:
            input = super().inverse_keypoints(input, params, extra_args=extra_args)
        return input

    def inverse(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Inverse transformation.

        Used to inverse a tensor according to the performed transformation by a forward pass, or with respect to
        provided parameters.
        """
        if params is None:
            if self._params is not None:
                params = self._params
            else:
                raise RuntimeError("No valid params to inverse the transformation.")

        return self.inverse_inputs(input, params, extra_args=extra_args)

    def forward(
        self, input: Tensor, params: Optional[List[ParamItem]] = None, extra_args: Dict[str, Any] = {}
    ) -> Tensor:
        """Define the video computation performed."""
        if len(input.shape) != 5:
            raise AssertionError(f"Input must be a 5-dim tensor. Got {input.shape}.")

        if params is None:
            self._params = self.forward_parameters(input.shape)
            params = self._params

        output = self.transform_inputs(input, params, extra_args=extra_args)

        return output
