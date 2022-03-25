Augmentation Containers
=======================

.. currentmodule:: kornia.augmentation.container

The classes in this section are containers for augmenting different data formats (e.g. images, videos).


Augmentation Sequential
-----------------------

Kornia augmentations provides simple on-device augmentation framework with the support of various syntax sugars
(e.g. return transformation matrix, inverse geometric transform). Therefore, we provide advanced augmentation
container to ease the pain of building augmenation pipelines. This API would also provide predefined routines
for automating the processing of masks, bounding boxes, and keypoints.

.. autoclass:: AugmentationSequential

   .. automethod:: forward

   .. automethod:: inverse



ImageSequential
---------------

Kornia augmentations provides simple on-device augmentation framework with the support of various syntax sugars
(e.g. return transformation matrix, inverse geometric transform). Additionally, ImageSequential supports the
mix usage of both image processing and augmentation modules.

.. autoclass:: ImageSequential

   .. automethod:: forward


PatchSequential
---------------

.. autoclass:: PatchSequential

   .. automethod:: forward


Video Data Augmentation
-----------------------

Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which can be referred as 2.5D than 3D.
In most applications, augmenting video data requires a static temporal dimension to have the same augmentations are performed for each frame.
Thus, `VideoSequential` can be used to do such trick as same as `nn.Sequential`.
Currently, `VideoSequential` supports data format like :math:`(B, C, T, H, W)` and :math:`(B, T, C, H, W)`.

.. code-block:: python

   import kornia.augmentation as K

   transform = K.VideoSequential(
      K.RandomAffine(360),
      K.ColorJiggle(0.2, 0.3, 0.2, 0.3),
      data_format="BCTHW",
      same_on_frame=True
   )

.. autoclass:: VideoSequential

   .. automethod:: forward
