from kornia.augmentation._2d.mix.transplantation import RandomTransplantation
from kornia.augmentation._3d.base import AugmentationBase3D

__all__ = ["RandomTransplantation3D"]


class RandomTransplantation3D(RandomTransplantation, AugmentationBase3D):  # type: ignore
    """RandomTransplantation3D augmentation.

    3D version of the :class:`kornia.augmentation.RandomTransplantation` augmentation intended to be used with
    :class:`kornia.augmentation.AugmentationSequential`. The interface is identical to the 2D version.
    """

    pass
