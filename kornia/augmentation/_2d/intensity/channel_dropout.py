from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor


class RandomChannelDropout(IntensityAugmentationBase2D):
    r"""Apply random dropout to channels of a batch of multi-dimensional images.

    Args:
        same_on_batch: Apply the same transformation across the batch.
        p: Probability of applying the transformation to each channel.
        keepdim: Whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.arange(1 * 2 * 2 * 2.).view(1, 2, 2, 2)
        >>> RandomChannelDropout()(img)
        tensor([[[[0., 1.],
                  [2., 3.]],
        <BLANKLINE>
                 [[0., 0.],
                  [0., 0.]]]])

    To apply the exact augmentation again, you may take advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomChannelDropout(p=0.5)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, C, _, _ = shape
        dropout_mask = torch.rand(B, C) > self.p

        return {"dropout_mask": dropout_mask.to(torch.float32)}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        dropout_mask = params["dropout_mask"].unsqueeze(2).unsqueeze(3).expand_as(input)

        return input * dropout_mask
