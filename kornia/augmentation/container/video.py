from typing import cast, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import kornia
from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase, TensorWithTransformMat
from kornia.augmentation.container.base import SequentialBase

from .image import ImageSequential, ParamItem

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

        >>> input, label = torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1), torch.tensor([0, 1])
        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ... random_apply=10,
        ... data_format="BCTHW",
        ... same_on_frame=True)
        >>> output = aug_list(input)
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(True)
        >>> (output[0, :, 1] == output[0, :, 2]).all()
        tensor(True)
        >>> (output[0, :, 2] == output[0, :, 3]).all()
        tensor(True)

        If set `same_on_frame` to False:

        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.augmentation.RandomMixUp(p=1.0),
        ... data_format="BCTHW",
        ... same_on_frame=False)
        >>> output, lab = aug_list(input)
        >>> output.shape, lab.shape
        (torch.Size([2, 3, 4, 5, 6]), torch.Size([2, 4, 3]))
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(False)

        Reproduce with provided params.
        >>> out2, lab2 = aug_list(input, label, params=aug_list._params)
        >>> torch.equal(output, out2)
        True
    """

    def __init__(
        self,
        *args: nn.Module,
        data_format: str = "BTCHW",
        same_on_frame: bool = True,
        random_apply: Union[int, bool, Tuple[int, int]] = False,
    ) -> None:
        super(VideoSequential, self).__init__(
            *args, same_on_batch=None, return_transform=None, keepdim=None, random_apply=random_apply
        )
        self.same_on_frame = same_on_frame
        self.data_format = data_format.upper()
        assert self.data_format in ["BCTHW", "BTCHW"], f"Only `BCTHW` and `BTCHW` are supported. Got `{data_format}`."
        self._temporal_channel: int
        if self.data_format == "BCTHW":
            self._temporal_channel = 2
        elif self.data_format == "BTCHW":
            self._temporal_channel = 1

    def __infer_channel_exclusive_batch_shape__(self, batch_shape: torch.Size, chennel_index: int) -> torch.Size:
        # Fix mypy complains: error: Incompatible return value type (got "Tuple[int, ...]", expected "Size")
        return cast(torch.Size, batch_shape[:chennel_index] + batch_shape[chennel_index + 1 :])

    def __repeat_param_across_channels__(self, param: torch.Tensor, frame_num: int) -> torch.Tensor:
        """Repeat parameters across channels.

        The input is shaped as (B, ...), while to output (B * same_on_frame, ...), which
        to guarentee that the same transformation would happen for each frame.

        (B1, B2, ..., Bn) => (B1, ... B1, B2, ..., B2, ..., Bn, ..., Bn)
                              | ch_size | | ch_size |  ..., | ch_size |
        """
        repeated = param[:, None, ...].repeat(1, frame_num, *([1] * len(param.shape[1:])))
        return repeated.reshape(-1, *list(param.shape[1:]))  # type: ignore

    def _input_shape_convert_in(
        self, input: torch.Tensor, label: Optional[torch.Tensor], frame_num: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Convert any shape to (B, T, C, H, W)
        if self.data_format == "BCTHW":
            # Convert (B, C, T, H, W) to (B, T, C, H, W)
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass

        if label is not None:
            if label.shape == input.shape[:2]:
                # if label is provided as (B, T)
                label = label.view(-1)
            elif label.shape == input.shape[:1]:
                label = label[..., None].repeat(1, frame_num).view(-1)
            elif label.shape == torch.Size([input.shape[0] * input.shape[1]]):
                # Skip the conversion if label is provided as (B * T,)
                pass
            else:
                raise NotImplementedError(f"Invalid label shape of {label.shape}.")
        input = input.reshape(-1, *input.shape[2:])
        return input, label

    def _input_shape_convert_back(
        self, input: torch.Tensor, label: Optional[torch.Tensor], frame_num: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input = input.view(-1, frame_num, *input.shape[1:])
        if self.data_format == "BCTHW":
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass

        if label is not None:
            label = label.view(input.size(0), frame_num, -1)
        return input, label

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
            if isinstance(module, (SequentialBase,)):
                seq_param = module.forward_parameters(batch_shape)
                if self.same_on_frame:
                    raise ValueError("Sequential is currently unsupported for ``same_on_frame``.")
                param = ParamItem(name, seq_param)
            elif isinstance(module, (_AugmentationBase, MixAugmentationBase)):
                mod_param = module.forward_parameters(batch_shape)
                if self.same_on_frame:
                    for k, v in mod_param.items():
                        # TODO: revise colorjitter order param in the future to align the standard.
                        if not (k == "order" and isinstance(module, kornia.augmentation.ColorJitter)):
                            mod_param.update({k: self.__repeat_param_across_channels__(v, frame_num)})
                param = ParamItem(name, mod_param)
            else:
                param = ParamItem(name, None)
            params.append(param)
        return params

    def forward(  # type: ignore
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None, params: Optional[List[ParamItem]] = None
    ) -> Union[TensorWithTransformMat, Tuple[TensorWithTransformMat, torch.Tensor]]:
        """Define the video computation performed."""
        assert len(input.shape) == 5, f"Input must be a 5-dim tensor. Got {input.shape}."

        if params is None:
            params = self.forward_parameters(input.shape)

        # Size of T
        frame_num = input.size(self._temporal_channel)
        input, label = self._input_shape_convert_in(input, label, frame_num)

        out = super().forward(input, label, params)  # type: ignore
        if self.return_label:
            output, label = cast(Tuple[TensorWithTransformMat, torch.Tensor], out)
        else:
            output = cast(TensorWithTransformMat, out)

        if isinstance(output, (tuple, list)):
            _out, label = self._input_shape_convert_back(output[0], label, frame_num)
            output = (_out, output[1])
        else:
            output, label = self._input_shape_convert_back(output, label, frame_num)

        return self.__packup_output__(output, label)
