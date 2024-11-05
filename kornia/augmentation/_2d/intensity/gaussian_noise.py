from typing import Any, Dict, Optional, Tuple

import torch

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor


def _randn_like(input: Tensor, mean: float, std: float) -> Tensor:
    x = torch.randn_like(input)  # Generating on GPU is fastest with `torch.randn_like(...)`
    if std != 1.0:  # `if` is cheaper than multiplication
        x *= std
    if mean != 0.0:  # `if` is cheaper than addition
        x += mean
    return x


class RandomGaussianNoise(IntensityAugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    .. image:: _static/img/RandomGaussianNoise.png

    Args:
        mean: The mean of the gaussian distribution.
        std: The standard deviation of the gaussian distribution.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> RandomGaussianNoise(mean=0., std=1., p=1.)(img)
        tensor([[[[ 2.5410,  0.7066],
                  [-1.1788,  1.5684]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGaussianNoise(mean=0., std=1., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"mean": mean, "std": std}

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        return {}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if "gaussian_noise" in params:
            gaussian_noise = params["gaussian_noise"]
        else:
            gaussian_noise = _randn_like(input, mean=flags["mean"], std=flags["std"])
            self._params["gaussian_noise"] = gaussian_noise
        return input + gaussian_noise
