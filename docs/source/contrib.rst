kornia.contrib
==============

.. meta::
   :description: The kornia.contrib module provides contributed image operations, such as sliding-window patches, connected-component labelling, distance transforms, histogram matching, plasma-fractal noise, k-means clustering and augmentation audits, and task-level wrappers of edge, face and object detection, super-resolution, image stitching and prompted segmentation models.

.. currentmodule:: kornia.contrib

Contributed image operations, and task-level wrappers that pair a kornia model with its pre- and post-processing.
The image operations:

- Sliding-window patches: :func:`extract_tensor_patches` cuts them, :func:`combine_tensor_patches` puts them back
  and averages overlaps, and :func:`compute_padding` gives the padding that makes the windows fit.
- Connected components: :func:`connected_components` labels them with a fixed number of pooling steps and
  :func:`connected_components_union_find` exactly; :ref:`labeling-method` compares the two.
- Distances to the nearest positive pixel: :func:`distance_transform`.
- Histogram matching to a template: :func:`histogram_matching`, built on the 1-D interpolation :func:`interp`.
- Plasma-fractal noise maps: :func:`diamond_square`.
- K-means clustering: :class:`KMeans`.
- A Python callable as a module: :class:`Lambda`.
- The geometry of one augmentation pipeline call: :func:`audit`, described in :ref:`augmentation-audit`.

.. _augmentation-audit:

Augmentation Geometry Audit
---------------------------

``kornia.contrib.augmentation_audit.audit`` executes an
``AugmentationSequential`` once and returns ``(outputs, report)``. It records the
executed operation order, native parameter snapshots, image shapes, effective
flags, and coordinate matrices. It also checks keypoint and box round trips against
the labels actually returned by the pipeline. The ordinary ``forward`` path has no
reporting overhead.

For example, a training-data validation step can inspect coordinates and export a
report without storing image pixels::

    import json
    from pathlib import Path
    import torch
    import kornia.augmentation as K
    from kornia.contrib import audit

    image = torch.rand(2, 3, 32, 48)
    points = torch.tensor([[[4., 6.], [20., 15.]]]).expand(2, -1, -1)
    pipeline = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        K.RandomAffine(degrees=15, p=1.0),
        data_keys=["input", "keypoints"],
    )
    (image_out, points_out), report = audit(
        pipeline, image, points, roundtrip_tolerance=1e-3, out_of_frame_tolerance=0.25,
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
  ``RandomCrop`` prepadding when it belongs to the returned image mapping.
  ``inverse_matrix`` is its algebraic inverse, with nonfinite entries for batches
  that cannot be inverted. ``invertible`` identifies those batches explicitly.
* ``geometry_status="available"`` describes matrix availability rather than spatial
  alignment. Round trips inverse-map the returned labels and compare them with the
  source. An invertible matrix can therefore accompany a nonzero label error.
* Keypoint errors are Euclidean distances. Box errors are symmetric Hausdorff
  distances between corner sets. Tensor box exports take axis-aligned envelopes,
  so rotation or shear can introduce measurable loss; ``Boxes`` objects retain the
  transformed quadrilateral.
* Per-batch statistics include only finite errors. Inspect
  ``roundtrip_valid_count`` together with the mean, median, or maximum. Empty labels
  and unavailable inverses do not produce a successful zero-error measurement.
* Out-of-frame counts use inclusive pixel centers from zero to ``width - 1`` and
  ``height - 1``. A box is outside when any corner is outside. Nonfinite labels are
  counted separately.
* Crop matrices can invert coordinates even after image content has been discarded.
  The report records this distinction and never evaluates image reconstruction.
  Cropping and downsampling emit possible content-loss warnings.
* A ``RandomCrop`` skipped for the whole batch leaves images and labels unchanged,
  including with ``padding`` or ``pad_if_needed`` (#4473). Manually supplied
  parameters that apply a shape-changing operation to only some rows raise
  ``ValueError``, as the ordinary forward call does, because the skipped rows
  cannot keep their shape (#4497). With ``p=1.0, p_batch=1.0``, an overridden
  ``batch_prob`` is not consulted: every image row is cropped and the labels move
  with it (#5585). Pass ``p < 1`` to skip rows by hand.
* Non-rigid and unknown operations are explicitly unsupported for matrix
  composition. A transformation-matrix identity fallback is not treated as proof
  of correspondence. Supported neighboring operations remain in the provenance.
* Captured operations must match the ordered operations selected by the sampled
  parameters. Missing or out-of-order captures make the geometry unsupported.
  ``RandAugment``, ``AutoAugment``, ``TrivialAugment``, and custom sequential
  subclasses are currently opaque and therefore unsupported when selected.

Diagnostics are computed in float64 for a float64 image and in float32 otherwise,
and disable autocast for matrix composition and inverse mapping. The pipeline
forward itself inherits the caller's autocast context, so returned labels remain
identical to an ordinary forward call. Autocast can quantize spatial-label
propagation and produce real round-trip errors; the report does not suppress
those differences.

The API accepts one nonempty BCHW image first, followed by optional batched masks,
keypoints, or boxes in the usual positional ``data_keys`` forms. Dictionaries,
ragged or unbatched inputs, video, patch, and 3D containers, and modules registered
under multiple names are unsupported. Repeated sampled operations and nested
built-in 2D sequences are supported. Spatial operations must preserve label shapes
and cardinality. This is an eager diagnostic utility and the same stateful pipeline
must not be audited concurrently.

``params`` is an independent, detached native replay snapshot. JSON is a portable
diagnostic export rather than a replay loader. Tensor exports include dtype, device,
and shape metadata; nonfinite values become JSON ``null``.

.. autofunction:: audit

.. autoclass:: AugmentationAuditReport
   :members: summary, to_dict, to_json

.. autoclass:: AugmentationAuditStep

.. autoclass:: SpatialAudit

Models
------

Base
^^^^
.. autoclass:: kornia.models.base.ModelBase
    :members:
    :undoc-members:

EfficientViT
^^^^^^^^^^^^

.. autoclass:: kornia.models.efficient_vit.EfficientViT
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.models.efficient_vit.EfficientViTConfig
    :members:
    :undoc-members:

Backbones
^^^^^^^^^

.. autoclass:: kornia.models.efficient_vit.backbone.EfficientViTBackbone
    :members:
    :undoc-members:

.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_b0
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_b1
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_b2
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_b3

.. autoclass:: kornia.models.efficient_vit.backbone.EfficientViTLargeBackbone
    :members:
    :undoc-members:

.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_l0
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_l1
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_l2
.. autofunction:: kornia.models.efficient_vit.backbone.efficientvit_backbone_l3

Structures
^^^^^^^^^^

.. _anchor SegmentationResults:
.. autoclass:: kornia.models.structures.SegmentationResults
    :members:
    :undoc-members:

.. autoclass:: kornia.models.structures.Prompts
    :members:
    :undoc-members:

VisualPrompter
--------------

.. autoclass:: kornia.contrib.visual_prompter.VisualPrompter
    :members: set_image, reset_image, compile, predict, preprocess_image, preprocess_prompts

Edge Detection
--------------

.. autoclass:: EdgeDetector

Face Detection
--------------

.. autoclass:: FaceDetector

.. autoclass:: FaceKeypoint
    :members:
    :undoc-members:

.. autoclass:: FaceDetectorResult
    :members:
    :undoc-members:

Object Detection
----------------

.. autoclass:: kornia.contrib.object_detection.BoundingBoxDataFormat
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.contrib.object_detection.BoundingBox
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.object_detection.ObjectDetectorResult
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.object_detection.ObjectDetector
    :members:
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.contrib.object_detection.ResizePreProcessor
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.object_detection.BoxFiltering
    :members: forward, to_onnx

.. autofunction:: kornia.contrib.object_detection.results_from_detections

Real-Time Detection Transformer (RT-DETR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: kornia.models.rt_detr.RTDETRModelType
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.models.rt_detr.RTDETRConfig
    :members:
    :undoc-members:

.. autoclass:: kornia.models.rt_detr.RTDETR
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.models.rt_detr.DETRPostProcessor
    :members:
    :undoc-members:

Image Segmentation
------------------
.. autofunction:: connected_components

.. autofunction:: connected_components_union_find

.. _labeling-method:

Choosing a labeling method
^^^^^^^^^^^^^^^^^^^^^^^^^^

Both functions label 8-connected components: two foreground pixels (pixels equal
to 1) that touch at an edge or a corner belong to one component. SciPy, OpenCV
and scikit-image take any non-zero pixel as foreground, so the same partition is
``scipy.ndimage.label(mask == 1, structure=np.ones((3, 3)))`` (SciPy's default
structure is 4-connected), ``cv2.connectedComponents((mask == 1).astype(np.uint8),
connectivity=8)`` (OpenCV's default connectivity) and
``skimage.measure.label(mask == 1, connectivity=2)`` (scikit-image's default for
2-D images). Label values differ between these libraries and between the two
functions: compare foreground partitions, not label values.

``connected_components_union_find`` computes the exact 8-connected partition
with no iteration budget, so long structures such as vessels, roads and text
strokes come out whole. Its convergence checks synchronize the device with the
host, so it does not compile with ``torch.compile(fullgraph=True)`` and cannot
be captured in a CUDA graph.

``connected_components`` runs a fixed ``num_iterations`` of 3x3 max-pooling
steps, with no host synchronization, and compiles with ``fullgraph=True``. On
CUDA, 10 steps run several times faster than union-find; on the CPU, union-find
is faster than 10 steps.

Both return ``int64`` labels, 0 on the background, and neither is
differentiable.

Segment Anything (SAM)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: kornia.models.sam.SamModelType
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.models.sam.SamConfig
    :members:
    :undoc-members:

.. autoclass:: kornia.models.sam.Sam
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

Image Patches
-------------

.. autofunction:: compute_padding
.. autofunction:: extract_tensor_patches
.. autofunction:: combine_tensor_patches

.. autoclass:: ExtractTensorPatches
.. autoclass:: CombineTensorPatches

Histogram Matching
------------------

.. autofunction:: histogram_matching
.. autofunction:: interp

Image Classification
--------------------

.. autoclass:: kornia.models.vit.VisionTransformer
    :members:
.. autoclass:: kornia.models.vit_mobile.MobileViT
.. autoclass:: TinyViT
    :members:

Image Stitching
---------------

.. autoclass:: ImageStitcher

Lambda
------

.. autoclass:: Lambda

Distance Transform
------------------

.. autofunction:: distance_transform

.. autoclass:: DistanceTransform

Plasma Fractal Noise
--------------------

.. autofunction:: diamond_square

KMeans
------

.. autoclass:: KMeans
    :members:

Super Resolution
----------------

.. autoclass:: SuperResolution
    :members:

.. autoclass:: SuperResolutionConfig
    :members:

.. autoclass:: SmallSRBuilder
    :members:

.. autoclass:: RRDBNetBuilder
    :members:
