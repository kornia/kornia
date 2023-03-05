from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomChannelDropout(IntensityAugmentationBase2D):
    """
    Randomly Drop Channels in the input Image.

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
        self.fill_value = fill_value
        KORNIA_CHECK(
            isinstance(channel_drop_range, tuple) and len(channel_drop_range) == 2,
            "Invalid channel_drop_range. Should be tuple of length 2.",
        )
        KORNIA_CHECK(
            1 <= channel_drop_range[0] <= channel_drop_range[1],
            "Invalid channel_drop_range. Max channel should be greater than lower.",
        )
        self.min_channel = channel_drop_range[0]
        self.max_channel = channel_drop_range[1]
        print(channel_drop_range)

        num_channels_to_drop = torch.randint(low=self.min_channel, 
                                                high=(self.max_channel + 1), 
                                                    size=(1,))[0]
        self.channels_to_drop = self.generate_parameters(batch_shape=(0, self.max_channel - 1, num_channels_to_drop))[
            "channel_params"
        ].tolist()  # tolist to fix typing tests
        
    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        params = super().generate_parameters(batch_shape)
        # +1 to avoid possible error, when low = high
        params["channel_params"] = torch.randint(low=batch_shape[0], high=batch_shape[1] + 1, size=(batch_shape[2],))

        return params

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK(
            (input.shape[-3] != 1 and len(input.shape) != 2), "Only one channel. ChannelDropout is not defined."
        )
        num_channels = input.shape[-3]
        KORNIA_CHECK(self.max_channel < num_channels, "Can not drop all channels in ChannelDropout.")

        input[..., self.channels_to_drop, :, :] = self.fill_value

        return input
