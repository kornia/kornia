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

   .. automethod:: audit


Auditing geometric provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AugmentationSequential.audit`` executes the ordinary pipeline once and returns
``(outputs, report)``. It records captured operation order (including repeated
operations and nested built-in 2D sequences), parameters, image shapes, effective
flags (including image-call keyword overrides) and coordinate matrices, and checks
capture completeness against the sampled execution parameters. The returned
outputs have the same structure and gradients
as ``forward``. Reporting has additional snapshot and diagnostic costs; ordinary
``forward`` does not enable it.

For example, a training-data validation step can inspect coordinates and export
an artifact without storing image pixels::

    import json
    from pathlib import Path
    import torch
    import kornia.augmentation as K

    image = torch.rand(2, 3, 32, 48)
    points = torch.tensor([[[4., 6.], [20., 15.]]]).expand(2, -1, -1)
    pipeline = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        K.RandomAffine(degrees=15, p=1.0),
        data_keys=["input", "keypoints"],
    )
    (image_out, points_out), report = pipeline.audit(
        image, points, roundtrip_tolerance=1e-3, out_of_frame_tolerance=0.25,
    )
    print(report.summary())
    point_audit = report.spatial[0]
    assert report.geometry_status == "available"
    assert torch.equal(point_audit.roundtrip_valid_count, point_audit.count)
    assert bool((point_audit.roundtrip_max <= 1e-3).all())
    Path("augmentation-audit.json").write_text(report.to_json(), encoding="utf-8")
    assert json.loads(report.to_json())["image_reconstruction_evaluated"] is False

    # Native parameter snapshots replay the run with this pipeline configuration.
    replay_image, replay_points = pipeline(image, points, params=report.params)
    torch.testing.assert_close(replay_image, image_out)
    torch.testing.assert_close(replay_points, points_out)

Interpreting the report
^^^^^^^^^^^^^^^^^^^^^^^

* ``matrix`` maps original pixel coordinates to final image coordinates, including
  ``RandomCrop`` prepadding when it is part of the returned image's mapping.
  ``inverse_matrix`` is its algebraic inverse, with
  nonfinite entries for batches that cannot be inverted. ``invertible`` records
  those batches explicitly. Half-precision diagnostics use float32.
* ``geometry_status="available"`` describes matrix availability, not successful
  spatial alignment. Round trips inverse-map the **actual returned** labels and
  compare them with the source. A pipeline can expose an invertible matrix yet
  have incorrect spatial propagation, which yields nonzero errors.
* Keypoint errors are Euclidean distances. Box errors are symmetric Hausdorff
  distances between corner sets, so corner ordering does not affect the result.
  Tensor box exports take axis-aligned envelopes; rotation/shear can therefore
  introduce measurable round-trip loss. Passing a ``Boxes`` object retains the
  transformed quadrilateral instead.
* Per-batch statistics include only finite errors. Always inspect
  ``roundtrip_valid_count`` as well as the mean, median or maximum; empty labels
  and unavailable inverses do not have a successful zero-error measurement.
* Out-of-frame counts use inclusive pixel centers, from zero to ``width - 1``
  and ``height - 1``. A box is outside when any corner is outside. Nonfinite
  labels have a separate count. Fractions describe the supplied labels, **not**
  image-area coverage or mask alignment.
* Crop matrices can invert coordinates even though the crop discarded image
  content. The report records this distinction and never evaluates image
  reconstruction. A decrease in image height or width also produces a possible
  sampling/content-loss warning, including for resize operations. These warnings
  do not cover every kind of information loss: interpolation and intensity
  operations can also lose pixels without changing image dimensions.
* Partially applied ``RandomCrop`` mappings follow the returned image branch,
  not just the label coordinates. A skipped crop can currently leave an image
  unchanged while padding its labels; its nonzero round-trip error indicates a
  real inconsistency and is not suppressed. Shape-changing slice crops with
  unapplied rows have no reliable cached matrix for every returned image row,
  so their geometry is reported as ``unsupported``.
* Non-rigid and unknown operations are explicitly unsupported for matrix
  composition. The ``silent`` transformation-matrix mode's identity fallback
  is not treated as evidence of valid correspondence. Supported neighboring
  operations still appear in the per-operation provenance.
* Captured operations must match the ordered, repeated operations selected by
  this call's native parameters. If a selected operation bypasses forward hooks,
  geometry is ``unsupported`` even when other operations were captured; the report
  does not invent its shape or matrix. Unselected operations do not count as
  missing. A truly empty pipeline can report an identity mapping.
* ``RandAugment``, ``AutoAugment`` and ``TrivialAugment`` policies, and custom
  sequential-container subclasses, are currently opaque to the audit. A selected
  opaque container makes geometry ``unsupported``; its internal operations are
  not certified. The ordinary pipeline outputs are still returned.

The API accepts one nonempty BCHW image first, with optional batched masks,
keypoints and boxes in the usual positional ``data_keys`` forms. Dictionaries,
ragged/unbatched inputs, video, patch and 3D augmentation containers, and a module
registered under multiple names are not supported. Sampling the same registered
operation repeatedly with ``random_apply`` is supported. The API is an eager
diagnostic tool, not a compiled/exported graph operation; the same stateful
pipeline should not be audited concurrently.
Spatial operations must preserve label shapes and cardinality; the audit rejects
changed shapes instead of broadcasting unmatched labels into a comparison.

The report records operation flags and ``configured_extra_args`` (including mask
resampling overrides), but does not evaluate whether a mask's interpolation is
suitable for its label semantics. It does not claim that a seed alone reproduces
an unrecorded random state. ``params`` is an independent, detached native replay
snapshot; JSON is a portable diagnostic export, not a replay loader. Tensor
exports include dtype, device and shape metadata; nonfinite values become JSON
``null``. Report tensors are isolated from later forwards but remain mutable
PyTorch tensors.

.. autoclass:: AugmentationAuditReport
   :members: summary, to_dict, to_json

.. autoclass:: AugmentationAuditStep

.. autoclass:: SpatialAudit


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
