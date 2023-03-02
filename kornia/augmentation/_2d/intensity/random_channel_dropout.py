from typing import Any, Dict, Optional, Union, Tuple

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomChannelDropout(IntensityAugmentationBase2D):
    """Randomly Drop Channels in the input Image.
    Args:
        channel_drop_range: range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)` or :math:`(C, H, W)`
    Examples:
        >>> _ =torch.random.manual_seed(1)
        >>> img = torch.ones(1, 3, 5, 5)
        >>> out = RandomChannelDropout(p=1)(img)
        >>> out.shape
        torch.Size([1, 3, 5, 5])
        >>> out
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                 [[1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.],
                  [1., 1., 1., 1., 1.]]]])
    """

    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: Union[int, float] = 0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value
        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]
        KORNIA_CHECK(
            1 <= self.min_channels <= self.max_channels, f"Invalid channel_drop_range. Got: {channel_drop_range}"
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK(
            not (
                (len(input.shape) == 3 and input.shape[0] == 1)
                or (len(input.shape) == 4 and input.shape[1] == 1)
                or (len(input.shape) == 2)
            ),
            "Only one channel. ChannelDropout is not defined.",
        )
        num_channels = input.shape[-3]
        KORNIA_CHECK(self.max_channels < num_channels, "Can not drop all channels in ChannelDropout.")
        num_drop_channels = rg.PlainUniformGenerator(((self.min_channels, self.max_channels), "factor_1", None, None))(
            torch.Size([1])
        )["factor_1"]

        channels_to_drop = rg.PlainUniformGenerator(((0, num_channels), "factor_1", None, None))(
            torch.Size([int(torch.round(num_drop_channels).item())])
        )["factor_1"]
        channels_to_drop = list(map(lambda t: t.to(torch.int), channels_to_drop))
        input_cp = input.clone()
        input_cp[..., channels_to_drop, :, :] = self.fill_value
        return input_cp
