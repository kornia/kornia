Local features (detector and descriptor together)
=================================================

.. currentmodule:: kornia.feature

:class:`LocalFeature` runs a detector and a LAF descriptor in one call, and the SIFT and HardNet presets below are
its subclasses. Each preset returns LAFs ``(B, N, 2, 3)`` in pixels, their responses ``(B, N)`` and descriptors
``(B, N, D)``, with ``N = num_features`` slots for every image. The slots no detection filled hold a zero LAF and
share one descriptor, so drop them with :func:`laf_is_filled` before matching; :class:`LocalFeatureMatcher` does it
for you. :class:`SIFTFeature`, :class:`KeyNetHardNet` and :class:`KeyNetAffNetHardNet` take one image per call,
``(1, 1, H, W)``; the other presets take a batch of grayscale images ``(B, 1, H, W)``, and
``SIFTFeatureScaleSpace(descriptor_backend="pyramid")`` also converts an RGB batch to grayscale. The learned models,
:class:`ALIKED`, :class:`DeDoDe`, :class:`DISK`, :class:`XFeat` and :class:`SOLD2_detector`, are not
:class:`LocalFeature` subclasses; :ref:`learned-local-features` gives their inputs and outputs.

.. _sift-pipelines:

Choosing a SIFT pipeline
------------------------

For new scale-space SIFT pipelines, start with
:class:`SIFTFeatureScaleSpace` and explicitly select ``descriptor_backend="pyramid"``::

    import kornia.feature as KF

    feature = KF.SIFTFeatureScaleSpace(
        num_features=2048, descriptor_backend="pyramid", rootsift=True
    ).to(grayscale_image.device)
    lafs, responses, descriptors = feature(grayscale_image)

``grayscale_image`` is a floating-point tensor of shape ``(B, 1, H, W)`` with values
in ``[0, 1]``. Outputs are LAFs ``(B, N, 2, 3)``, responses ``(B, N)`` and descriptors
``(B, N, 128)``. ``N`` is the requested feature budget; fewer detections leave padded
slots. Use :func:`laf_is_filled` to identify real detections before matching.
``rootsift=True`` selects RootSIFT normalization. Use ``descriptor_layout="opencv"``
when feeding descriptors to :class:`LightGlueMatcher` with ``"sift"``.

Both feature classes default to ``descriptor_backend="patch"``. Backend performance
depends on device, image size, feature count and batching; the recipe is not a claim
that pyramid extraction is always faster.

.. list-table:: SIFT pipeline choices
   :header-rows: 1
   :widths: 25 45 30

   * - Pipeline
     - Detection and extraction
     - When to use it
   * - ``SIFTFeatureScaleSpace(..., descriptor_backend="pyramid")``
     - Dedicated DoG scale-space detector; orientation and description share the
       detector's Gaussian layers.
     - Primary recipe for new scale-space SIFT use.
   * - ``SIFTFeatureScaleSpace(..., descriptor_backend="patch")``
     - Generic :class:`ScaleSpaceDetector`; orientation and description operate on
       resampled patches.
     - Existing pipelines, or workloads where measurements favor patch extraction.
   * - ``SIFTFeature(..., descriptor_backend="patch")``
     - Multiresolution, per-level DoG detection; patch orientation and description.
     - Existing multiresolution pipelines; this is an approximation to scale-space SIFT.
   * - ``SIFTFeature(..., descriptor_backend="pyramid")``
     - The same multiresolution detector; a separate image pyramid supplies smoothed
       gradients for orientation and pooled histogram maps for description.
     - Measured multiresolution workloads; see the validation below.

The two feature classes have different detector algorithms. :class:`SIFTFeature`
applies a single-scale DoG response independently at each image resolution, with
2-D non-maximum suppression. It does not select extrema across scale or refine
positions and scales to subpixel precision. :class:`SIFTFeatureScaleSpace` searches
3-D DoG scale-space extrema and refines their positions and scales. Neither preset
implements classical SIFT's contrast and Hessian edge rejection, and each assigns
only one orientation per detection (or preserves upright frames when requested).

``descriptor_backend`` is not an interchangeable descriptor switch. In
:class:`SIFTFeature` it changes orientation and description but keeps detection;
in :class:`SIFTFeatureScaleSpace` it also selects a different detector implementation.
Consequently, the scale-space backends can return different detections as well as
different descriptors.

The scale-space pyramid path ranks valid refined extrema by absolute response.
Precise doubling and integer octave decimation keep pixel coordinates aligned.
The converged integer Gaussian layer and refined continuous scale are retained
separately. Gradients are computed once per used layer and shared between orientation
and description; no pyramid is retained between calls. Sampling and Gaussian support
differ from patch extraction, so descriptors are not numerically interchangeable.
For this backend, ``compile_modules`` accepts ``scale_pyr`` and ``subpix`` (or
``True`` for both); sparse orientation and description run eagerly. The patch
backend passes ``compile_modules`` to :class:`ScaleSpaceDetector`.

Matching with LightGlue
^^^^^^^^^^^^^^^^^^^^^^^

Use ``descriptor_layout="opencv"`` with :class:`LightGlueMatcher` in ``"sift"``
mode. Pass the oriented LAFs returned by the feature, filter zero-LAF padding from
both LAFs and descriptor rows with :func:`laf_is_filled`, and supply the original
image sizes as ``hw1=(height0, width0)`` and ``hw2=(height1, width1)``. The matcher
converts the LAF angle and scale internally; do not apply those conversions twice.

The public adapter uses one sixth of the LAF scale. For the scale-space presets'
6-sigma frames this is the detection sigma. :class:`SIFTFeature` uses a different
measurement-region convention, so its multiresolution frames do not imply the same
detection sigma. Descriptor-layout compatibility alone does not guarantee equal
matching accuracy across the four pipelines. Evaluate the complete feature/matcher
combination at the intended feature budget.

Validation and possible deprecation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The automatic orientation used by ``SIFTFeature(descriptor_backend="pyramid")``
integrates a 19x19 support of smoothed image gradients before angular binning,
without extracting image patches. Its acceptance criteria, tracked in
`#5748 <https://github.com/kornia/kornia/issues/5748>`_, include performance at
1024 features and Oxford accuracy at 2048 features. Promotion of this convenience
combination requires accuracy matching the patch backend and a demonstrated
runtime or memory advantage on a stated workload.
Validation should compare with the patch backend at 1024 and 2048 features on
IMC PhotoTourism pose estimation and Oxford homography estimation, reporting
matcher settings, device and timing methodology.

If validation does not establish those benefits, the plan is to deprecate this
convenience combination and remove it only after the :doc:`API stability policy
</get-started/stability>` window: at least one minor release of warnings and a
documented replacement. No deprecation warning or removal version is set.
This plan concerns the combination, not the standalone
:class:`SIFTDescriptorFromPyramid` descriptor-only interface; see
:ref:`sift-descriptor-interfaces`.

HardNet presets
---------------

The six HardNet presets describe 32 x 32 patches with a pretrained :class:`HardNet`, through
:class:`LAFDescriptor`. They load every network they use with its pretrained weights when they are built and come
back in evaluation mode. They differ in the detector, in whether :class:`LAFAffNetShapeEstimator` adapts the
affine shape of each frame before its orientation, in the orientation estimator, which ``upright=True`` skips, and
in the images they take:

.. list-table:: HardNet presets
   :header-rows: 1
   :widths: 22 34 14 16 14

   * - Preset
     - Detector and frames
     - Orientation
     - Input, ``num_features``
     - ``LightGlueMatcher`` mode
   * - :class:`GFTTAffNetHardNet`
     - :class:`ScaleSpaceDetector` on :class:`CornerGFTT` corners, AffNet shapes
     - ``LAFOrienter(19)``
     - ``(B, 1, H, W)``, 8000
     - none
   * - :class:`HesAffNetHardNet`
     - :class:`ScaleSpaceDetector` on :class:`BlobHessian` blobs, AffNet shapes
     - ``LAFOrienter(19)``
     - ``(B, 1, H, W)``, 2048
     - none
   * - :class:`DoGHardNet`
     - The DoG detector of ``SIFTFeatureScaleSpace(descriptor_backend="patch")``, frames scaled to 12 sigma
     - ``LAFOrienter(19)``
     - ``(B, 1, H, W)``, 2048
     - ``"doghardnet"``
   * - :class:`DoGAffNetHardNet`
     - The same, with AffNet shapes
     - ``LAFOrienter(19)``
     - ``(B, 1, H, W)``, 2048
     - ``"dog_affnet_hardnet"``
   * - :class:`KeyNetHardNet`
     - :class:`KeyNetDetector`
     - :class:`OriNet`
     - ``(1, 1, H, W)`` in ``[0, 1]``, 8000
     - none
   * - :class:`KeyNetAffNetHardNet`
     - :class:`KeyNetDetector`, AffNet shapes
     - :class:`OriNet`
     - ``(1, 1, H, W)`` in ``[0, 1]``, 8000
     - ``"keynet_affnet_hardnet"``

For a preset with a :class:`LightGlueMatcher` mode, drop the padded slots and pass its LAFs and descriptors to the
matcher in that mode. The DoG presets return the 12-sigma frames that the ``"doghardnet"`` weights expect.

.. _learned-local-features:

Learned local features
----------------------

:class:`ALIKED`, :class:`DISK`, :class:`XFeat` and :class:`DeDoDe` detect and describe keypoints with learned
networks, and :class:`SOLD2_detector` detects line segments. Load the first four with their ``from_pretrained``,
which returns the model in evaluation mode: their constructors build random weights in training mode, although
``DeDoDe()`` with its default ``G`` descriptor already loads a pretrained DINOv2 backbone. ``SOLD2_detector()``
loads its pretrained weights. The four return keypoints as ``(x, y)`` pixels, not LAFs (except
:meth:`ALIKED.forward_laf`), and ``laf_from_center_scale_ori(keypoints[None])`` (:func:`laf_from_center_scale_ori`)
turns the keypoints of one image into the LAFs that :class:`LightGlueMatcher` takes in the mode of the table. Call
the matcher on the outputs of :class:`XFeat`, which are inference tensors, under ``torch.no_grad()`` or
``torch.inference_mode()`` (`#6533 <https://github.com/kornia/kornia/issues/6533>`_). The
:doc:`image matching guide </applications/image_matching>` compares the speed and accuracy of the four keypoint
models with the authors' code, device by device.

.. list-table:: Learned local features
   :header-rows: 1
   :widths: 14 30 38 18

   * - Model
     - Input
     - Output
     - ``LightGlueMatcher`` mode
   * - :class:`ALIKED`
     - RGB ``(B, 3, H, W)`` in ``[0, 1]``, or grayscale ``(B, 1, H, W)``, of any size
     - one :class:`ALIKEDFeatures` per image: sub-pixel keypoints and 128-D descriptors (64-D for ``"aliked-t16"``);
       :meth:`ALIKED.forward_laf` pads them into LAF tensors
     - ``"aliked"`` (not for ``"aliked-t16"``)
   * - :class:`DISK`
     - RGB ``(B, 3, H, W)``, sides multiples of 16 unless ``pad_if_not_divisible=True``
     - one :class:`DISKFeatures` per image: integer-pixel keypoints in raster order and 128-D descriptors
     - ``"disk"``
   * - :class:`XFeat`
     - ``(B, C, H, W)``, any channel count, resized to multiples of 32
     - :meth:`XFeat.detectAndCompute`: one dict per image, keypoints and 64-D descriptors;
       :meth:`XFeat.match_xfeat` and :meth:`XFeat.match_xfeat_star` also match image pairs
     - ``"xfeat"``
   * - :class:`DeDoDe`
     - RGB ``(B, 3, H, W)`` in ``[0, 1]``, padded to multiples of 14
     - tensors of the ``n`` best keypoints of each image, ``(B, N, 2)`` with pixel ``i`` at ``i + 0.5``, their
       scores and 256-D descriptions
     - ``"dedodeb"``, ``"dedodeg"``
   * - :class:`SOLD2_detector`
     - grayscale ``(B, 1, H, W)`` in ``[0, 1]``, sides multiples of 8
     - line segments, one ``(N, 2, 2)`` tensor of ``(row, col)`` endpoints per image; :class:`SOLD2` also describes
       and matches them
     - none

API reference
-------------

.. autoclass:: LocalFeature
   :members: forward

.. autoclass:: ALIKED
   :members: forward, from_pretrained, forward_laf

.. autoclass:: ALIKEDFeatures
   :undoc-members:
   :members: n, to

.. autoclass:: DeDoDe
   :members: forward, from_pretrained, describe, detect

.. autoclass:: DISK
   :members: forward, from_pretrained, heatmap_and_dense_descriptors

.. autoclass:: DISKFeatures
   :undoc-members:
   :members: x, y, to

.. autoclass:: XFeat
   :members: from_pretrained, detectAndCompute, detectAndComputeDense, match_xfeat, match_xfeat_star

.. autoclass:: XFeatModel
   :members: forward

.. autoclass:: InterpolateSparse2d
   :members: forward

.. autoclass:: SOLD2_detector
   :members: forward

.. autoclass:: SIFTFeatureScaleSpace
   :members: forward

.. autoclass:: SIFTFeature
   :members: forward

.. autoclass:: GFTTAffNetHardNet
   :members: forward

.. autoclass:: HesAffNetHardNet
   :members: forward

.. autoclass:: DoGHardNet
   :members: forward

.. autoclass:: DoGAffNetHardNet
   :members: forward

.. autoclass:: KeyNetAffNetHardNet
   :members: forward

.. autoclass:: KeyNetHardNet
   :members: forward
