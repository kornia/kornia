Augmentation Containers
=======================

.. meta::
   :description: The Augmentation Containers module in Kornia provides advanced frameworks for building augmentation pipelines. It includes classes like AugmentationSequential, ManyToManyAugmentationDispatcher, and VideoSequential for managing data formats such as images, videos, and temporal data. It also supports processing masks, bounding boxes, and keypoints in augmentation workflows.

.. currentmodule:: kornia.augmentation.container

The classes in this section are containers for augmenting different data formats (e.g. images, videos).


Augmentation Sequential
-----------------------

Kornia augmentations provide a simple on-device augmentation framework with a number of conveniences
(e.g. returning the transformation matrix, or inverting a geometric transform). On top of that, we provide an
advanced augmentation container to ease the pain of building augmentation pipelines. This API also provides
predefined routines that automate the processing of masks, bounding boxes, and keypoints.

.. autoclass:: AugmentationSequential

   .. automethod:: forward

   .. automethod:: inverse


Augmentation Dispatchers
------------------------
Kornia supports two types of augmentation dispatching, namely many-to-many and many-to-one. The former wraps
different augmentations into one group and lets the user pass as many inputs as there are augmentations, applying
each augmentation to the corresponding input. The latter applies different augmentations to a single input in order
to obtain a list of differently transformed outputs.

.. note::
   The class names below keep their historical spelling (``Dispather``) for backward compatibility.

.. autoclass:: ManyToManyAugmentationDispather

   .. automethod:: forward


.. autoclass:: ManyToOneAugmentationDispather

   .. automethod:: forward



ImageSequential
---------------

``ImageSequential`` is a lightweight container that, in addition to augmentation modules, accepts arbitrary
image processing ``nn.Module`` instances (e.g. the modules in :mod:`kornia.filters`, :mod:`kornia.color` or
:mod:`kornia.enhance`), so both kinds of operations can be mixed in a single pipeline.

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
  modules (Gaussian blur, edge detection, color transforms).
- A lightweight container is preferred without the overhead of multi-target
  synchronization logic.

Example using ``ImageSequential``::

    import torch
    import kornia.augmentation as K
    from kornia.augmentation.container import ImageSequential
    from kornia.filters import GaussianBlur2d

    img = torch.rand(1, 3, 256, 256)

    seq = ImageSequential(
        K.RandomHorizontalFlip(p=1.0),
        GaussianBlur2d((3, 3), (1.5, 1.5)),  # any differentiable nn.Module can be inserted
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

Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which is
sometimes referred to as 2.5D rather than 3D. In most applications, augmenting video data requires the same
augmentation, with the same parameters, to be applied to every frame of a clip. `VideoSequential` does exactly that,
with the same interface as `nn.Sequential`. It supports the :math:`(B, C, T, H, W)` and :math:`(B, T, C, H, W)`
data formats.

.. code-block:: python

   import torch
   import kornia.augmentation as K

   transform = K.VideoSequential(
      K.RandomAffine(360),
      K.ColorJiggle(0.2, 0.3, 0.2, 0.3),
      data_format="BCTHW",
      same_on_frame=True,
   )
   clip = torch.rand(2, 3, 8, 64, 64)  # 2 clips of 8 RGB frames
   out = transform(clip)

.. autoclass:: VideoSequential

   .. automethod:: forward
