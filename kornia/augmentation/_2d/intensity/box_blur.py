from typing import Any, Dict, Optional, Tuple

from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.filters import box_blur


class RandomBoxBlur(IntensityAugmentationBase2D):
    """Add random blur with a box filter to an image tensor.

    .. image:: _static/img/RandomBoxBlur.png

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``.
        normalized: if True, L1 norm of the kernel is set to 1.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.filters.box_blur`.

    Examples:
        >>> img = torch.ones(1, 1, 24, 24)
        >>> out = RandomBoxBlur((7, 7))(img)
        >>> out.shape
        torch.Size([1, 1, 24, 24])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomBoxBlur((7, 7), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (3, 3),
        border_type: str = "reflect",
        normalized: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"kernel_size": kernel_size, "border_type": border_type, "normalized": normalized}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return box_blur(input, flags["kernel_size"], flags["border_type"], flags["normalized"])
