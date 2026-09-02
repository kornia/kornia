2D geometric transforms
=======================

.. currentmodule:: kornia.augmentation

Transforms that move pixels. Each one exposes its ``(B, 3, 3)`` transformation matrix, can be inverted, and is
applied consistently to masks, boxes and keypoints by :class:`~kornia.augmentation.AugmentationSequential`.

.. autoclass:: CenterCrop
.. autoclass:: PadTo
.. autoclass:: RandomAffine
.. autoclass:: RandomCrop
.. autoclass:: RandomElasticTransform
.. autoclass:: RandomErasing
.. autoclass:: RandomFisheye
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomPerspective
.. autoclass:: RandomResizedCrop
.. autoclass:: RandomRotation90
.. autoclass:: RandomRotation
.. autoclass:: RandomShear
.. autoclass:: RandomThinPlateSpline
.. autoclass:: RandomVerticalFlip

Resize
------

Deterministic resizing operators, shape-agnostic for 2D and 3D tensors.

.. autoclass:: LongestMaxSize
.. autoclass:: Resize
.. autoclass:: SmallestMaxSize
