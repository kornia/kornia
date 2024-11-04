from typing import Any, Dict, Optional

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import normalize_min_max


class RandomAutoContrast(IntensityAugmentationBase2D):
    r"""Apply a random auto-contrast of a tensor image.

    Args:
        p: probability of applying the transformation.
        clip_output: if true clip output
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.normalize_min_max`
    """

    def __init__(
        self, clip_output: bool = True, same_on_batch: bool = False, p: float = 1.0, keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        self.clip_output = clip_output

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        out = normalize_min_max(input)

        if self.clip_output:
            return out.clamp(0.0, 1.0)

        return out
