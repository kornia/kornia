from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class RandomChannelDropout(IntensityAugmentationBase2D):
    r"""Drops channels in the input image.

    In case batch of images is passed, for each image random channels will be dropped.

    .. image:: _static/img/RandomChannelDropout.png

    Args:
        channel_drop_range: range from which we choose the number of channels to drop.
        fill_value: pixel value for the dropped channel.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.arange(1*2*2*2.).view(1,2,2,2)
        >>> RandomChannelDropout()(img)
        tensor([[[[0., 1.],
                  [2., 3.]],
        <BLANKLINE>
                 [[0., 0.],
                  [0., 0.]]]])
    """

    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        num_drop_channels: int = 1,
        fill_value: Union[int, float] = 0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.fill_value = fill_value
        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]
        KORNIA_CHECK(
            1 <= self.min_channels <= self.max_channels, f"Invalid channel_drop_range. Got: {channel_drop_range}"
        )

        self._param_generator = rg.PlainUniformGenerator(
            (
                num_drop_channels,
                "num_drop_channels",
                (self.max_channels - self.min_channels + 1) // 2,
                (self.min_channels, self.max_channels + 1),
            )
        )

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Tensor]:
        B, C, _, _ = batch_shape

        KORNIA_CHECK(C > 1, "Channel dropout can be applied only to multichannel images")
        KORNIA_CHECK(self.max_channels < C, "Cannot drop all channels in ChannelDropout.")

        params = super().generate_parameters(batch_shape)
        num_drop_channels = params["num_drop_channels"]

        channels_to_drop = torch.argsort(torch.rand(B, C))
        channels_to_drop_one_hot = torch.zeros((B, C), dtype=torch.uint8)

        # # set first k element in each row to 1 (for each row it is variable amount of them)
        channels_to_drop_one_hot = torch.where(
            (torch.arange(C)[None].repeat(B, 1) + num_drop_channels[:, None]) < C, 0, 1
        )

        channels_to_drop_one_hot = channels_to_drop_one_hot.gather(dim=1, index=channels_to_drop)

        params["channels_to_drop"] = channels_to_drop_one_hot

        return params

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
        _, C, _, _ = input.shape

        KORNIA_CHECK(C > 1, "Channel dropout can be applied only to multichannel images")
        KORNIA_CHECK(self.max_channels < C, "Cannot drop all channels in ChannelDropout.")

        input[params["channels_to_drop"] == 1, ...] = torch.tensor(
            self.fill_value, dtype=input.dtype, device=input.device
        )
        return input
