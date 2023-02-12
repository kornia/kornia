from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import posterize


class RandomPosterize(IntensityAugmentationBase2D):
    r"""Posterize given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomPosterize.png

    Args:
        p: probability of applying the transformation.
        bits: Integer that ranged from (0, 8], in which 0 gives black image and 8 gives the original.
            If int x, bits will be generated from (x, 8) then convert to int.
            If tuple (x, y), bits will be generated from (x, y) then convert to int.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.posterize`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> posterize = RandomPosterize(3., p=1.)
        >>> posterize(input)
        tensor([[[[0.4863, 0.7529, 0.0784, 0.1255, 0.2980],
                  [0.6275, 0.4863, 0.8941, 0.4549, 0.6275],
                  [0.3451, 0.3922, 0.0157, 0.1569, 0.2824],
                  [0.5176, 0.6902, 0.8000, 0.1569, 0.2667],
                  [0.6745, 0.9098, 0.3922, 0.8627, 0.4078]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomPosterize(3., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        bits: Union[float, Tuple[float, float], Tensor] = 3,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        # TODO: the generator should receive the device
        self._param_generator = rg.PosterizeGenerator(bits)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return posterize(input, params["bits_factor"].to(input.device))
