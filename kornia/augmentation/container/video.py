from typing import Tuple, Union, cast

import torch
import torch.nn as nn

import kornia
from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase


class VideoSequential(nn.Sequential):
    r"""VideoSequential for processing 5-dim video data like (B, T, C, H, W) and (B, C, T, H, W).

    `VideoSequential` is used to replace `nn.Sequential` for processing video data augmentations.
    By default, `VideoSequential` enabled `same_on_frame` to make sure the same augmentations happen
    across temporal dimension. Meanwhile, it will not affect other augmentation behaviours like the
    settings on `same_on_batch`, etc.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        data_format (str): only BCTHW and BTCHW are supported. Default: BTCHW.
        same_on_frame (bool): apply the same transformation across the channel per frame. Default: True.

    Example:
        If set `same_on_frame` to True, we would expect the same augmentation has been applied to each
        timeframe.

        >>> input = torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1)
        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
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
        ... data_format="BCTHW",
        ... same_on_frame=False)
        >>> output = aug_list(input)
        >>> output.shape
        torch.Size([2, 3, 4, 5, 6])
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(False)
    """

    def __init__(self, *args: _AugmentationBase, data_format="BTCHW", same_on_frame: bool = True) -> None:
        super(VideoSequential, self).__init__(*args)
        self.same_on_frame = same_on_frame
        for aug in args:
            if isinstance(aug, MixAugmentationBase):
                raise NotImplementedError(f"MixAugmentations are not supported at this moment. Got {aug}.")
        self.data_format = data_format.upper()
        assert self.data_format in ["BCTHW", "BTCHW"], \
            f"Only `BCTHW` and `BTCHW` are supported. Got `{data_format}`."
        self._temporal_channel: int
        if self.data_format == "BCTHW":
            self._temporal_channel = 2
        elif self.data_format == "BTCHW":
            self._temporal_channel = 1

    def __infer_channel_exclusive_batch_shape__(self, input: torch.Tensor) -> torch.Size:
        batch_shape: torch.Size = input.shape
        # Fix mypy complains: error: Incompatible return value type (got "Tuple[int, ...]", expected "Size")
        return cast(torch.Size, batch_shape[:self._temporal_channel] + batch_shape[self._temporal_channel + 1:])

    def __repeat_param_across_channels__(self, param: torch.Tensor, frame_num: int) -> torch.Tensor:
        """Repeat parameters across channels.

        The input is shaped as (B, ...), while to output (B * same_on_frame, ...), which
        to guarentee that the same transformation would happen for each frame.

        (B1, B2, ..., Bn) => (B1, ... B1, B2, ..., B2, ..., Bn, ..., Bn)
                              | ch_size | | ch_size |  ..., | ch_size |
        """
        repeated = param[:, None, ...].repeat(1, frame_num, *([1] * len(param.shape[1:])))
        return repeated.reshape(-1, *list(param.shape[1:]))  # type: ignore

    def _input_shape_convert_in(self, input: torch.Tensor) -> torch.Tensor:
        # Convert any shape to (B, T, C, H, W)
        if self.data_format == "BCTHW":
            # Convert (B, C, T, H, W) to (B, T, C, H, W)
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass
        return input

    def _input_shape_convert_back(self, input: torch.Tensor, frame_num: int) -> torch.Tensor:
        input = input.view(-1, frame_num, *input.shape[1:])
        if self.data_format == "BCTHW":
            input = input.transpose(1, 2)
        if self.data_format == "BTCHW":
            pass
        return input

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(input.shape) == 5, f"Input must be a 5-dim tensor. Got {input.shape}."
        # Size of T
        frame_num = input.size(self._temporal_channel)
        # Got param generation shape to (B, C, H, W). Ignoring T.
        batch_shape = self.__infer_channel_exclusive_batch_shape__(input)
        input = self._input_shape_convert_in(input)
        input = input.reshape(-1, *batch_shape[1:])
        if not self.same_on_frame:
            # Overwrite param generation shape to (B * T, C, H, W).
            batch_shape = input.shape
        for aug in self.children():
            aug = cast(_AugmentationBase, aug)
            param = aug.forward_parameters(batch_shape)
            if self.same_on_frame:
                for k, v in param.items():
                    # TODO: revise colorjitter order param in the future to align the standard.
                    if not (k == "order" and isinstance(aug, kornia.augmentation.ColorJitter)):
                        param.update({k: self.__repeat_param_across_channels__(v, frame_num)})
            input = aug(input, params=param)

        if isinstance(input, (tuple, list)):
            input[0] = self._input_shape_convert_back(input[0], frame_num)
        else:
            input = self._input_shape_convert_back(input, frame_num)

        return input
