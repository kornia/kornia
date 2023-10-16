from __future__ import annotations

from typing import Any, Optional

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import equalize_clahe


class RandomClahe(IntensityAugmentationBase2D):
    r"""Apply CLAHE equalization on the input tensor randomly.

    .. image:: _static/img/equalize_clahe.png

    Args:
        clip_limit: threshold value for contrast limiting. If 0 clipping is disabled.
        grid_size: number of tiles to be cropped in each direction (GH, GW).
        slow_and_differentiable: flag to select implementation
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    .. note::
        This function internally uses :func:`kornia.enhance.equalize_clahe`.

    Examples:
        >>> img = torch.rand(1, 10, 20)
        >>> aug = RandomClahe()
        >>> res = aug(img)
        >>> res.shape
        torch.Size([1, 1, 10, 20])

        >>> img = torch.rand(2, 3, 10, 20)
        >>> aug = RandomClahe()
        >>> res = aug(img)
        >>> res.shape
        torch.Size([2, 3, 10, 20])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomClahe(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        clip_limit: tuple[float, float] = (40.0, 40.0),
        grid_size: tuple[int, int] = (8, 8),
        slow_and_differentiable: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.clip_limit = clip_limit
        self._param_generator = rg.PlainUniformGenerator((self.clip_limit, "clip_limit_factor", None, None))
        self.flags = {"grid_size": grid_size, "slow_and_differentiable": slow_and_differentiable}

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        clip_limit = float(params["clip_limit_factor"][0])
        return equalize_clahe(input, clip_limit, flags["grid_size"], flags["slow_and_differentiable"])
