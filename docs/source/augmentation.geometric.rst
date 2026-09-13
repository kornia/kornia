2D geometric transforms
=======================

.. currentmodule:: kornia.augmentation

Spatial transformations and erasing operations. Matrix, inverse and annotation support depend on the class
and configuration. Slice-mode crops reject inversion; non-rigid transforms do not carry boxes or keypoints
along with the image (`#4420 <https://github.com/kornia/kornia/issues/4420>`_). `RandomErasing` changes image
and mask values without moving coordinates. See
:class:`~kornia.augmentation.AugmentationSequential` for the supported data-key paths and their limitations.

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
