2D geometric transforms
=======================

.. currentmodule:: kornia.augmentation

Spatial transforms, plus ``RandomErasing``, which changes image and mask values without moving coordinates. The
non-rigid ``RandomElasticTransform``, ``RandomThinPlateSpline`` and ``RandomFisheye`` cannot move boxes or
keypoints, so a container call that passes them raises; the last two raise for masks too. ``.inverse()`` leaves
these steps applied (`#4420 <https://github.com/kornia/kornia/issues/4420>`_). See
:class:`~kornia.augmentation.AugmentationSequential` for the supported data-key paths.

``RandomAffine`` combines rotation, translation, scale and shear in one warp; ``RandomPerspective`` warps
projectively. ``RandomCrop`` cuts a fixed-size patch at a random position, ``RandomResizedCrop`` cuts a patch of
random size and aspect ratio and resizes it, and ``CenterCrop`` always cuts the center. ``RandomRotation90``
rotates by multiples of 90 degrees.

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
.. autoclass:: RandomTranslate
.. autoclass:: RandomVerticalFlip

Resize
------

Deterministic resizing operators, shape-agnostic for 2D and 3D tensors.

.. autoclass:: LongestMaxSize
.. autoclass:: Resize
.. autoclass:: SmallestMaxSize
