from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.contrib import diamond_square
from kornia.core import Tensor


class RandomPlasmaBrightness(IntensityAugmentationBase2D):
    r"""Adds brightness to the image based on a fractal map generated by the diamond square algorithm.

    .. image:: _static/img/RandomPlasmaBrightness.png

    This is based on the original paper: TorMentor: Deterministic dynamic-path, data augmentations with fractals.
    See: :cite:`tormentor` for more details.

    .. note::
        This function internally uses :func:`kornia.contrib.diamond_square`.

    Args:
        roughness: value to scale during the recursion in the generation of the fractal map.
        intensity: value that scales the intensity values of the generated maps.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 3, 4)
        >>> RandomPlasmaBrightness(roughness=(0.1, 0.7), p=1.)(img)
        tensor([[[[0.6415, 1.0000, 0.3142, 0.6836],
                  [1.0000, 0.5593, 0.5556, 0.4566],
                  [0.5809, 1.0000, 0.7005, 1.0000]]]])
    """

    def __init__(
        self,
        roughness: Tuple[float, float] = (0.1, 0.7),
        intensity: Tuple[float, float] = (0.0, 1.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (roughness, "roughness", None, None), (intensity, "intensity", None, None)
        )

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = image.shape
        roughness = params["roughness"].to(image)
        intensity = params["intensity"].to(image).view(-1, 1, 1, 1)
        brightness_map = 2 * diamond_square((B, C, H, W), roughness, device=image.device, dtype=image.dtype) - 1
        brightness_map = brightness_map * intensity
        return (image + brightness_map).clamp_(0, 1)


class RandomPlasmaContrast(IntensityAugmentationBase2D):
    r"""Adds contrast to the image based on a fractal map generated by the diamond square algorithm.

    .. image:: _static/img/RandomPlasmaContrast.png

    This is based on the original paper: TorMentor: Deterministic dynamic-path, data augmentations with fractals.
    See: :cite:`tormentor` for more details.

    .. note::
        This function internally uses :func:`kornia.contrib.diamond_square`.

    Args:
        roughness: value to scale during the recursion in the generation of the fractal map.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 3, 4)
        >>> RandomPlasmaContrast(roughness=(0.1, 0.7), p=1.)(img)
        tensor([[[[0.9651, 1.0000, 1.0000, 1.0000],
                  [1.0000, 0.9103, 0.8038, 0.9263],
                  [0.6882, 1.0000, 0.9544, 1.0000]]]])
    """

    def __init__(
        self,
        roughness: Tuple[float, float] = (0.1, 0.7),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((roughness, "roughness", None, None))

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = image.shape
        roughness = params["roughness"].to(image)
        contrast_map = 4 * diamond_square((B, C, H, W), roughness, device=image.device, dtype=image.dtype)
        return ((image - 0.5) * contrast_map + 0.5).clamp_(0, 1)


class RandomPlasmaShadow(IntensityAugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    .. image:: _static/img/RandomPlasmaShadow.png

    This is based on the original paper: TorMentor: Deterministic dynamic-path, data augmentations with fractals.
    See: :cite:`tormentor` for more details.

    .. note::
        This function internally uses :func:`kornia.contrib.diamond_square`.

    Args:
        roughness: value to scale during the recursion in the generation of the fractal map.
        shade_intensity: value that scales the intensity values of the generated maps.
        shade_quantity: value to select the pixels to mask.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 3, 4)
        >>> RandomPlasmaShadow(roughness=(0.1, 0.7), p=1.)(img)
        tensor([[[[0.7682, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000, 1.0000]]]])
    """

    def __init__(
        self,
        roughness: Tuple[float, float] = (0.1, 0.7),
        shade_intensity: Tuple[float, float] = (-1.0, 0.0),
        shade_quantity: Tuple[float, float] = (0.0, 1.0),
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (roughness, "roughness", None, None),
            (shade_intensity, "shade_intensity", None, None),
            (shade_quantity, "shade_quantity", None, None),
        )

    def apply_transform(
        self, image: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        B, _, H, W = image.shape
        roughness = params["roughness"].to(image)
        shade_intensity = params["shade_intensity"].to(image).view(-1, 1, 1, 1)
        shade_quantity = params["shade_quantity"].to(image).view(-1, 1, 1, 1)
        shade_map = diamond_square((B, 1, H, W), roughness, device=image.device, dtype=image.dtype)
        shade_map = (shade_map < shade_quantity).to(image.dtype) * shade_intensity
        return (image + shade_map).clamp_(0, 1)
