3D transforms
=============

.. currentmodule:: kornia.augmentation

Operators for volumetric ``(B, C, D, H, W)`` tensors, for example medical volumes. Use
:class:`~kornia.augmentation.container.VideoSequential` to apply 2D transforms framewise to video.
The geometric transforms use voxel coordinates ordered ``(x, y, z)`` for tensors ordered ``(D, H, W)``. Use a
3D transform for volumes; the similarly named 2D transforms operate on images and do not accept a depth dimension.

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
