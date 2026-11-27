3D transforms
=============

.. currentmodule:: kornia.augmentation

Operators for volumetric ``(B, C, D, H, W)`` tensors, such as medical volumes. For video, apply 2D transforms
frame by frame with :class:`~kornia.augmentation.container.VideoSequential`. The geometric transforms order voxel
coordinates ``(x, y, z)``, the reverse of the tensor's ``(D, H, W)``. Inside
:class:`~kornia.augmentation.container.AugmentationSequential`, use :class:`RandomTransplantation3D` for volumes;
:class:`RandomTransplantation` handles volumes only when called directly.

Geometric
---------

.. autoclass:: CenterCrop3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomPerspective3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomVerticalFlip3D

Intensity
---------

.. autoclass:: RandomEqualize3D
.. autoclass:: RandomMotionBlur3D

Mix
---

.. autoclass:: RandomTransplantation3D
