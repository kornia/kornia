kornia.augmentation
===================

This module implements in a high level logic. The main features of this module, and similar to the rest of the
library, is that can it perform data augmentation routines in a batch mode, using any supported device,
and can be used for backpropagation. Some of the available functionalities which are worth to mention are the
following: random rotations; affine and perspective transformations; several random color intensities transformations,
image noise distortion, motion blurring, and many of the different differentiable data augmentation policies.
In addition, we include a novel feature which is not found in other augmentations frameworks,
which allows the user to retrieve the applied transformation or chained transformations after each
call e.g. the generated random rotation matrix which can be used later to undo the image transformation
itself, or to be applied to additional metadata such as the label images for semantic segmentation,
in bounding boxes or landmark keypoints for object detection tasks. It gives the user the flexibility to
perform complex data augmentations pipelines.


.. currentmodule:: kornia.augmentation

.. toctree::

   augmentation.base
   augmentation.module
   augmentation.container
