Augmentation Containers
=======================

.. meta::
   :description: The Augmentation Containers module in Kornia provides advanced frameworks for building augmentation pipelines. It includes classes like AugmentationSequential, ManyToManyAugmentationDispather, and VideoSequential for managing data formats such as images, videos, and temporal data. It also supports processing masks, bounding boxes, and keypoints in augmentation workflows.

.. currentmodule:: kornia.augmentation.container

Containers chain augmentations into one pipeline. Use :class:`AugmentationSequential` when masks, boxes or keypoints
must follow the image, and :class:`ImageSequential` for images alone. :class:`VideoSequential` applies image
transforms to every frame of a video, and :class:`PatchSequential` to every patch of an image grid. The dispatchers
run several pipelines, either all on one input or each on its own input.


Augmentation Sequential
-----------------------

``AugmentationSequential`` samples its parameters once per call and applies them to every input named in
``data_keys``: images, masks, boxes and keypoints. It records the draw for replay, composes the matrices of its
geometric steps, and undoes them with ``inverse()``. Inversion restores coordinates, not the pixels lost to cropping
or resampling, and it leaves color changes in place.

.. autoclass:: AugmentationSequential

   .. automethod:: forward

   .. automethod:: inverse


Augmentation Dispatchers
------------------------
Kornia supports two types of augmentation dispatching, namely many-to-many and many-to-one. The former wraps
different augmentations into one group and lets the user pass as many inputs as there are augmentations, applying
each augmentation to the corresponding input; a call with a different number of input bundles than
augmentations raises ``ValueError`` before any augmentation runs. The latter applies different augmentations to a
single input in order to obtain a list of differently transformed outputs. Its members must be
``AugmentationSequential`` instances. With the default ``strict=True``, their ``data_keys`` must match;
``strict=False`` disables that construction check.

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

``AugmentationSequential`` coordinates supported data-key handlers from one set
of recorded parameters. Its non-rigid, mask-list and dtype limitations still
apply; see the class's Convention block. ``ImageSequential`` processes image
tensors without dispatching auxiliary annotations.


PatchSequential
---------------

.. autoclass:: PatchSequential

   .. automethod:: forward


Video Data Augmentation
-----------------------

Video data is a special case of 3D volumetric data that contains both spatial and temporal information, which is
sometimes referred to as 2.5D rather than 3D. In most applications, augmenting video data requires the same
augmentation, with the same parameters, to be applied to every frame of a clip. `VideoSequential` does exactly that,
with the same interface as `nn.Sequential`. It supports the ``(B, C, T, H, W)`` and ``(B, T, C, H, W)``
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


Container building blocks
-------------------------

Base classes and parameter records for writing a custom container. To build a pipeline, use the containers above.

.. autoclass:: kornia.augmentation.container.base.BasicSequentialBase
.. autoclass:: kornia.augmentation.container.base.SequentialBase
.. autoclass:: kornia.augmentation.container.base.ImageSequentialBase
.. autoclass:: kornia.augmentation.container.params.ParamItem
.. autoclass:: kornia.augmentation.container.params.PatchParamItem


Data keys
---------

.. currentmodule:: kornia.constants

``data_keys`` of :class:`~kornia.augmentation.container.AugmentationSequential` accepts a :class:`DataKey` member or
its name, for example ``DataKey.INPUT`` or ``"input"``.

.. autoclass:: DataKey
   :members:
   :undoc-members:
