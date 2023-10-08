from typing import Any, Dict, Optional, Tuple
import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor

class SaltAndPepperNoise(IntensityAugmentationBase2D):
    r"""Add salt and pepper noise to a batch of multi-dimensional images.

    Args:
        salt_prob: Probability of adding salt noise to each pixel.
        pepper_prob: Probability of adding pepper noise to each pixel.
        same_on_batch: Apply the same transformation across the batch.
        p: Probability of applying the transformation.
        keepdim: Whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> SaltAndPepperNoise(salt_prob=0.01, pepper_prob=0.5, p=1.0)(img)
        tensor([[[[0., 1.],
                  [1., 1.]]]])
    """

    def __init__(
        self, salt_prob: float = 0.01, pepper_prob: float = 0.01, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"salt_prob": salt_prob, "pepper_prob": pepper_prob}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        salt_prob = flags["salt_prob"]
        pepper_prob = flags["pepper_prob"]
        noisy_image = input.clone()
        # Generate a random mask for salt pixels
        salt_mask = torch.rand_like(input) < salt_prob

        # Generate a random mask for pepper pixels
        pepper_mask = torch.rand_like(input) < pepper_prob

        # Set salt pixels to 1 (white)
        noisy_image[salt_mask] = 1.0

        # Set pepper pixels to 0 (black)
        noisy_image[pepper_mask] = 0.0

        return noisy_image
