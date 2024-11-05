from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.filters import median_blur


class RandomMedianBlur(IntensityAugmentationBase2D):
    """Add random blur with a median filter to an image tensor.

    .. image:: _static/img/RandomMedianBlur.png

    Args:
        kernel_size: the blurring kernel size.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.filters.median_blur`.

    Examples:

        >>> img = torch.ones(1, 1, 4, 4)
        >>> out = RandomMedianBlur((3, 3), p = 1)(img)
        >>> out.shape
        torch.Size([1, 1, 4, 4])
        >>> out
        tensor([[[[0., 1., 1., 0.],
                  [1., 1., 1., 1.],
                  [1., 1., 1., 1.],
                  [0., 1., 1., 0.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomMedianBlur((7, 7), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, kernel_size: Tuple[int, int] = (3, 3), same_on_batch: bool = False, p: float = 0.5, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"kernel_size": kernel_size}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return median_blur(input, flags["kernel_size"])
