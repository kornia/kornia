:orphan:

Image Augmentations
===================

.. meta::
   :description: The Image Augmentations module in Kornia provides a wide range of 2D and 3D data augmentation transforms. It includes intensity-based augmentations, geometric transformations, mix-based augmentations, and normalization operations for both 2D and 3D image tensors. Key functions include random color shifts, rotations, cropping, elastic transformations, and more.

.. currentmodule:: kornia.augmentation

The augmentation operators, grouped by what they change. All of them accept ``(B, C, H, W)`` float tensors in
``[0, 1]`` (``(B, C, D, H, W)`` for the 3D transforms), sample independent parameters per batch element, and can
be combined with :class:`~kornia.augmentation.AugmentationSequential`.

.. list-table::
   :widths: 30 70

   * - :doc:`augmentation.intensity`
     - 2D transforms that change pixel values but not their positions: color jitter, blur, noise, illumination, JPEG, weather effects.
   * - :doc:`augmentation.geometric`
     - 2D transforms that move pixels: crops, flips, affine, perspective, elastic, thin-plate-spline, erasing.
   * - :doc:`augmentation.mix`
     - 2D transforms that combine several images: CutMix, MixUp, Mosaic, Jigsaw, Transplantation.
   * - :doc:`augmentation.transforms3d`
     - Geometric, intensity and mix transforms for volumetric ``(B, C, D, H, W)`` data.
