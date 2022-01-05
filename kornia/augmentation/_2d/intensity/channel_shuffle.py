from typing import Dict, Optional

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


class RandomChannelShuffle(IntensityAugmentationBase2D):
    r"""Shuffle the channels of a batch of multi-dimensional images.

    .. image:: _static/img/RandomChannelShuffle.png

    Args:
        return_transform: if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.

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

    def __init__(
        self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )

    def generate_parameters(self, shape: torch.Size) -> Dict[str, torch.Tensor]:
        B, C, _, _ = shape
        channels = torch.rand(B, C).argsort(dim=1)
        return dict(channels=channels)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = torch.empty_like(input)
        for i in range(out.shape[0]):
            out[i] = input[i, params["channels"][i]]
        return out
