from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor


class RandomChannelShuffle(IntensityAugmentationBase2D):
    r"""Shuffle the channels of a batch of multi-dimensional images.

    .. image:: _static/img/RandomChannelShuffle.png

    Args:
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input ``True`` or broadcast it
          to the batch form ``False``.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.arange(1*2*2*2.).view(1,2,2,2)
        >>> RandomChannelShuffle()(img)
        tensor([[[[4., 5.],
                  [6., 7.]],
        <BLANKLINE>
                 [[0., 1.],
                  [2., 3.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomChannelShuffle(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, C, _, _ = shape
        channels = torch.rand(B, C).argsort(dim=1)
        return {"channels": channels}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        out = torch.empty_like(input)
        for i in range(out.shape[0]):
            out[i] = input[i, params["channels"][i]]
        return out
