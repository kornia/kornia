from typing import Any, Dict, Optional

from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.enhance import equalize


class RandomEqualize(IntensityAugmentationBase2D):
    r"""Equalize given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomEqualize.png

    Args:
        p: Probability to equalize an image.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.equalize`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> equalize = RandomEqualize(p=1.)
        >>> equalize(input)
        tensor([[[[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
                  [0.6341, 0.4901, 0.8964, 0.4556, 0.6323],
                  [0.3489, 0.4017, 0.0223, 0.1689, 0.2939],
                  [0.5185, 0.6977, 0.8000, 0.1610, 0.2823],
                  [0.6816, 0.9152, 0.3971, 0.8742, 0.4194]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomEqualize(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(self, same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return equalize(input)
