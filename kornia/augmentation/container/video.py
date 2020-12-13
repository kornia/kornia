from typing import List, Tuple, Union, cast

import torch
import torch.nn as nn

import kornia
from kornia.augmentation.base import _AugmentationBase, MixAugmentationBase
from kornia.augmentation import ColorJitter


class VideoSequential(nn.Sequential):
    r"""VideoSequential for processing video data (B, C, T, H, W).

    `VideoSequential` is used to replace `nn.Sequential` for processing video data augmentations.
    By default, `VideoSequential` enabled `same_on_frame` to make sure the same augmentations happen
    across temporal dimension. Meanwhile, it will not affect other augmentation behaviours like the
    settings on `same_on_batch`, etc.

    Args:
        *args (_AugmentationBase): a list of augmentation module.
        same_on_frame (bool): apply the same transformation across the channel per frame. Default: True.

    Example:
        If set `same_on_frame` to True, we would expect the same augmentation has been applied to each
        timeframe.

        >>> input = torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1)
        >>> aug_list = VideoSequential(
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
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
        ... same_on_frame=False)
        >>> output = aug_list(input)
        >>> output.shape
        torch.Size([2, 3, 4, 5, 6])
        >>> (output[0, :, 0] == output[0, :, 1]).all()
        tensor(False)
    """

    def __init__(self, *args: _AugmentationBase, same_on_frame: bool = True) -> None:
        super(VideoSequential, self).__init__(*args)
        self.same_on_frame = same_on_frame
        for aug in args:
            if isinstance(aug, MixAugmentationBase):
                raise NotImplementedError(f"MixAugmentations are not supported at this moment. Got {aug}.")

    def __infer_channel_exclusive_batch_shape__(self, input: torch.Tensor) -> torch.Size:
        batch_shape: torch.Size = input.shape
        # Fix mypy complains: error: Incompatible return value type (got "Tuple[int, ...]", expected "Size")
        return cast(torch.Size, batch_shape[:2] + batch_shape[3:])

    def __repeat_param_across_channels__(self, param: torch.Tensor, frame_num: int) -> torch.Tensor:
        """Repeat parameters across channels.

        The input is shaped as (B, ...), while to output (B * same_on_frame, ...), which
        to guarentee that the same transformation would happen for each frame.

        (B1, B2, ..., Bn) => (B1, ... B1, B2, ..., B2, ..., Bn, ..., Bn)
                              | ch_size | | ch_size |  ..., | ch_size |
        """
        return torch.stack([param] * frame_num).T.reshape(-1, *list(param.shape[1:]))  # type: ignore

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(input.shape) == 5, f"Input must be (B, C, T, H, W). Got {input.shape}."
        # Size of T
        frame_num = input.size(2)
        # Got param generation shape to (B, C, H, W). Ignoring T.
        batch_shape = self.__infer_channel_exclusive_batch_shape__(input)
        # Convert to (B, T, C, H, W)
        input = input.transpose(1, 2)
        _original_shape = input.shape
        input = input.reshape(-1, *batch_shape[1:])
        if not self.same_on_frame:
            # Overwrite param generation shape to (B * T, C, H, W).
            batch_shape = input.shape
        for aug in self.children():
            aug = cast(_AugmentationBase, aug)
            param = aug.__forward_parameters__(batch_shape, aug.p, aug.p_batch, aug.same_on_batch)
            if self.same_on_frame:
                for k, v in param.items():
                    # TODO: revise colorjitter order param in the future to align the standard.
                    if not (k == "order" and isinstance(aug, ColorJitter)):
                        param.update({k: self.__repeat_param_across_channels__(v, frame_num)})
            input = aug(input, params=param)

        if isinstance(input, (tuple, list)):
            input[0] = input[0].view(*_original_shape).transpose(1, 2)
        else:
            input = input.view(*_original_shape).transpose(1, 2)

        return input
