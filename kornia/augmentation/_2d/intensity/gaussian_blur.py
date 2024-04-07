from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import BorderType
from kornia.filters import gaussian_blur2d


class RandomGaussianBlur(IntensityAugmentationBase2D):
    r"""Apply gaussian blur given tensor image or a batch of tensor images randomly. The standard deviation is
    sampled for each instance.

    .. image:: _static/img/RandomGaussianBlur.png

    Args:
        kernel_size: the size of the kernel.
        sigma: the range for the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``constant``, ``reflect``, ``replicate`` or ``circular``.
        separable: run as composition of two 1d-convolutions.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        silence_instantiation_warning: if True, silence the warning at instantiation.

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
        tensor([[[[0.5941, 0.5833, 0.5022, 0.4384, 0.3934],
                  [0.5310, 0.4964, 0.4113, 0.3637, 0.3472],
                  [0.4991, 0.4997, 0.4312, 0.3620, 0.3081],
                  [0.6082, 0.5667, 0.4954, 0.3825, 0.3508],
                  [0.7042, 0.6849, 0.6275, 0.4753, 0.4105]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        sigma: Union[Tuple[float, float], Tensor],
        border_type: str = "reflect",
        separable: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.flags = {
            "kernel_size": kernel_size,
            "separable": separable,
            "border_type": BorderType.get(border_type),
        }
        self._param_generator = rg.RandomGaussianBlurGenerator(sigma)

        self._gaussian_blur2d_fn = gaussian_blur2d

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        sigma = params["sigma"].unsqueeze(-1).expand(-1, 2)
        return self._gaussian_blur2d_fn(
            input,
            kernel_size=self.flags["kernel_size"],
            sigma=sigma,
            border_type=self.flags["border_type"].name.lower(),
            separable=self.flags["separable"],
        )

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: Optional[str] = None,
        options: Optional[Dict[Any, Any]] = None,
        disable: bool = False,
    ) -> "RandomGaussianBlur":
        self._gaussian_blur2d_fn = torch.compile(
            self._gaussian_blur2d_fn,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
        return self
