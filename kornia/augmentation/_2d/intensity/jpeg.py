from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.enhance import jpeg_codec_differentiable


class RandomJPEG(IntensityAugmentationBase2D):
    r"""Applies random (differentiable) JPEG coding to a tensor image.

    .. image:: _static/img/RandomJPEG.png

    Args:
        jpeg_quality: The range of compression rates to be applied.
        p: probability of applying the transformation.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.enhance.jpeg_codec_differentiable`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> images = 0.1904 * torch.ones(2, 3, 32, 32)
        >>> aug = RandomJPEG(jpeg_quality=(1.0, 50.0), p=1.)
        >>> images_jpeg = aug(images)

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> images = 0.1904 * torch.ones(2, 3, 32, 32)
        >>> aug = RandomJPEG(jpeg_quality=20.0, p=1.)  # Samples a JPEG quality from the range [30.0, 70.0]
        >>> (aug(images) == aug(images, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        jpeg_quality: Union[Tensor, float, Tuple[float, float], List[float]] = 50.0,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.jpeg_quality = jpeg_quality
        self._param_generator = rg.JPEGGenerator(jpeg_quality)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        jpeg_output: Tensor = jpeg_codec_differentiable(input, params["jpeg_quality"])
        return jpeg_output
