Augmentation Containers
=======================

.. meta::
   :name: description
   :content: "The Augmentation Containers module in Kornia provides advanced frameworks for building augmentation pipelines. It includes classes like AugmentationSequential, ManyToManyAugmentationDispatcher, and VideoSequential for managing data formats such as images, videos, and temporal data. It also supports processing masks, bounding boxes, and keypoints in augmentation workflows."

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


Augmentation Dispatchers
------------------------
Kornia supports two types of augmentation dispatching, namely many-to-many and many-to-one. The former wraps
different augmentations into one group and allows user to input multiple inputs in align with the number of
augmentations. The latter aims at performing different augmentations for one input that to obtain a list of
various transformed data.

.. autoclass:: ManyToManyAugmentationDispather

   .. automethod:: forward


.. autoclass:: ManyToOneAugmentationDispather

   .. automethod:: forward



ImageSequential
---------------

Kornia augmentations provides simple on-device augmentation framework with the support of various syntax sugars
(e.g. return transformation matrix, inverse geometric transform). Additionally, ImageSequential supports the
mix usage of both image processing and augmentation modules.

.. autoclass:: ImageSequential

   .. automethod:: forward

Differences Between ImageSequential and AugmentationSequential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ImageSequential`` and ``AugmentationSequential`` are both pipeline containers
in Kornia, but they're designed for fundamentally different data handling
scenarios. Understanding when to use each prevents common pitfalls in vision
pipelines.

**Use ``AugmentationSequential`` when:**

- The task requires synchronized transformations across multiple related tensors
  (images, masks, bounding boxes, keypoints).
- Spatial correspondence must be maintained between inputs and targets, as in
  semantic segmentation or object detection workflows.
- Multiple data formats need to be handled automatically with consistent random
  parameter sampling across all targets.

**Use ``ImageSequential`` when:**

- The pipeline only processes image tensors without auxiliary spatial targets.
- The workflow combines augmentation modules with general image processing
  operations (gaussian blur, edge detection, color transforms).
- A lightweight container is preferred without the overhead of multi-target
  synchronization logic.

Example using ``ImageSequential``::

    import torch
    import kornia.augmentation as K
    from kornia.augmentation.container import ImageSequential
    from kornia.filters import gaussian_blur2d

    img = torch.rand(1, 3, 256, 256)

    seq = ImageSequential(
        K.RandomHorizontalFlip(p=1.0),
        gaussian_blur2d,  # arbitrary differentiable ops can be inserted
    )

    out = seq(img)

Example using ``AugmentationSequential`` with synchronized transforms::

    import torch
    import kornia.augmentation as K

    img = torch.rand(1, 3, 256, 256)
    mask = torch.rand(1, 1, 256, 256)

    aug = K.AugmentationSequential(
        K.RandomResizedCrop((128, 128), p=1.0),
        data_keys=["input", "mask"],
    )

    img_out, mask_out = aug(img, mask)
    # identical random parameters applied to both tensors

The core distinction: ``AugmentationSequential`` guarantees that random
augmentation parameters are shared across all specified data keys, maintaining
geometric consistency. ``ImageSequential`` applies operations independently to
single image tensors without multi-target awareness.


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
