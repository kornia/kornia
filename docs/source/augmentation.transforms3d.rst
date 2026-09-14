3D transforms
=============

.. currentmodule:: kornia.augmentation

Operators for volumetric ``(B, C, D, H, W)`` tensors, e.g. medical volumes or video treated as 3D data.

Geometric
---------

.. autoclass:: CenterCrop3D
.. autoclass:: RandomAffine3D
.. autoclass:: RandomCrop3D
.. autoclass:: RandomDepthicalFlip3D
.. autoclass:: RandomHorizontalFlip3D
.. autoclass:: RandomRotation3D
.. autoclass:: RandomVerticalFlip3D

Intensity
---------

.. autoclass:: RandomEqualize3D
.. autoclass:: RandomMotionBlur3D

Mix
---

.. autoclass:: RandomTransplantation3D
