from typing import Dict, Optional, Tuple

from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import BorderType
from kornia.filters import gaussian_blur2d


class RandomGaussianBlur(IntensityAugmentationBase2D):
    r"""Apply gaussian blur given tensor image or a batch of tensor images randomly.

    .. image:: _static/img/RandomGaussianBlur.png

    Args:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.filters.gaussian_blur2d`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 5, 5)
        >>> blur = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)
        >>> blur(input)
        tensor([[[[0.6699, 0.4645, 0.3193, 0.1741, 0.1955],
                  [0.5422, 0.6657, 0.6261, 0.6527, 0.5195],
                  [0.3826, 0.2638, 0.1902, 0.1620, 0.2141],
                  [0.6329, 0.6732, 0.5634, 0.4037, 0.2049],
                  [0.8307, 0.6753, 0.7147, 0.5768, 0.7097]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = "reflect",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.flags = dict(kernel_size=kernel_size, sigma=sigma, border_type=BorderType.get(border_type))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        return gaussian_blur2d(
            input, self.flags["kernel_size"], self.flags["sigma"], self.flags["border_type"].name.lower()
        )
