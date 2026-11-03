2D geometric transforms
=======================

.. currentmodule:: kornia.augmentation

Spatial transformations and erasing operations. Matrix, inverse and annotation support depend on the class
and configuration. The non-rigid `RandomElasticTransform`, `RandomThinPlateSpline` and `RandomFisheye` cannot
move boxes and keypoints with the image, so a forward call with either raises; ``.inverse()`` leaves these steps
applied (`#4420 <https://github.com/kornia/kornia/issues/4420>`_). `RandomErasing` changes image
and mask values without moving coordinates. See
:class:`~kornia.augmentation.AugmentationSequential` for the supported data-key paths and their limitations.

Choose ``RandomAffine`` for one sampled rotation, translation, scale and shear; choose ``RandomPerspective`` for a
projective warp. ``RandomCrop`` preserves a sampled patch size, while ``RandomResizedCrop`` rescales that patch and
``CenterCrop`` is deterministic. The horizontal and vertical flips act on one image axis; ``RandomRotation90``
restricts rotations to quarter turns.

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
