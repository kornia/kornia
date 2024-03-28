from typing import Any, Dict, Optional

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import ChannelDropoutGenerator
from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE


class RandomChannelDropout(IntensityAugmentationBase2D):
    r"""Apply random channel dropout to a batch of images.

    .. image:: _static/img/RandomChannelDropout.png

    Args:
        num_drop_channels: Number of channels to drop randomly. Default is 1.
        fill_value: Value to fill the dropped channels with. Default is 0.0.
        same_on_batch: Apply the same transformation across the batch. Defaults to False.
        p: Probability of applying the transformation. Defaults to 0.5.
        keepdim: Whether to keep the output shape the same as input ``True`` or broadcast it
            to the batch form ``False``. Defaults to False.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(C, H, W)` or :math:`(B, C, H, W)`

    .. note::
        If `num_drop_channels` is set to 1, it means that for each image in the batch,
            we will randomly choose one channel to drop.
        If `num_drop_channels` is set to 2, it means that for each image in the batch,
            we will randomly choose two channels to drop.
        If num_drop_channels is set to 3, it means that for each image in the batch,
            we will randomly choose three channels to drop (all image).

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> img = torch.ones(1, 3, 3, 3)
        >>> aug = RandomChannelDropout(num_drop_channels=1, fill_value=0.0, p=1.0)
        >>> aug(img)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomChannelDropout(num_drop_channels=1, fill_value=0.0, p=1.0)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        num_drop_channels: int = 1,
        fill_value: float = 0.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        KORNIA_CHECK_TYPE(fill_value, float, f"`fill_value` must be a float. Got: {type(fill_value)}")
        KORNIA_CHECK(
            0.0 <= fill_value <= 1.0,
            f"Invalid value in `fill_value`. Must be a float between 0 and 1. Got: {fill_value}",
        )
        self.fill_value = tensor(fill_value)

        KORNIA_CHECK_TYPE(num_drop_channels, int, f"`num_drop_channels` must be an int. Got: {type(num_drop_channels)}")
        KORNIA_CHECK(
            num_drop_channels >= 1,
            f"Invalid value in `num_drop_channels`. Must be an int greater than 1. Got: {num_drop_channels}",
        )
        self.num_drop_channels = num_drop_channels
        # Generator of random parameters.
        self._param_generator = ChannelDropoutGenerator(self.num_drop_channels)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
        KORNIA_CHECK(
            self.num_drop_channels <= input.shape[1],
            "Invalid value in `num_drop_channels`. Cannot be greater than the number of channels of `input`.",
        )

        out = input.clone()
        out[params["batch_idx"], params["channel_idx"], ...] = self.fill_value.to(
            device=input.device, dtype=input.dtype
        )

        return out
