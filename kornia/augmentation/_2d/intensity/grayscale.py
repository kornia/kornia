from typing import Any, Dict, Optional

import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.color import rgb_to_grayscale


class RandomGrayscale(IntensityAugmentationBase2D):
    r"""Apply random transformation to Grayscale according to a probability p value.

    .. image:: _static/img/RandomGrayscale.png

    Args:
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
        p: probability of the image to be transformed to grayscale.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.color.rgb_to_grayscale`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn((1, 3, 3, 3))
        >>> aug = RandomGrayscale(p=1.0)
        >>> aug(inputs)
        tensor([[[[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGrayscale(p=1.0)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, rgb_weights: Optional[Tensor] = None, same_on_batch: bool = False, p: float = 0.1, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.rgb_weights = rgb_weights

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # Make sure it returns (*, 3, H, W)
        grayscale = torch.ones_like(input)
        grayscale[:] = rgb_to_grayscale(input, rgb_weights=self.rgb_weights)
        return grayscale
