Local features (detector and descriptor together)
=================================================

.. currentmodule:: kornia.feature

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

This is an explicit recipe: both feature classes still default to
``descriptor_backend="patch"``. No constructor default or numerical behavior is
changed by this recommendation. Backend performance depends on device, image size,
feature count and batching; the recipe is not a claim that pyramid extraction is
always faster.

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
     - Its automatic orientation was repaired in #5762. Use it for measured
       multiresolution workloads; #5748 remains open for the validation below.

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
backend retains the generic detector's compilation options.

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

`#5762 <https://github.com/kornia/kornia/pull/5762>`_ repaired the automatic
orientation used by ``SIFTFeature(descriptor_backend="pyramid")``. It integrates
a 19x19 support of smoothed image gradients before angular binning, without
extracting image patches. The issue tracked in
`#5748 <https://github.com/kornia/kornia/issues/5748>`_ remains open: its
acceptance criteria include performance at 1024 features and Oxford accuracy at
2048 features. Promotion of this convenience combination requires recovered
accuracy and a demonstrated runtime or memory advantage on a stated workload.
Validation should compare with the patch backend at 1024 and 2048 features on
IMC PhotoTourism pose estimation and Oxford homography estimation, reporting
matcher settings, device and timing methodology.

If repair does not establish those benefits, the plan is to deprecate this
convenience combination and remove it only after the :doc:`API stability policy
</get-started/stability>` window: at least one minor release of warnings and a
documented replacement. No formal deprecation warning or removal version is
introduced here. This plan concerns the combination, not the standalone
:class:`SIFTDescriptorFromPyramid` descriptor-only interface; see
:ref:`sift-descriptor-interfaces`.

API reference
-------------

.. autoclass:: LocalFeature
   :members: forward

.. autoclass:: SOLD2_detector
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

.. autoclass:: XFeat
   :members: forward, from_pretrained, detectAndCompute, detectAndComputeDense, match_xfeat, match_xfeat_star

.. autoclass:: XFeatModel
   :members: forward

.. autoclass:: InterpolateSparse2d
   :members: forward

.. autoclass:: DISKFeatures
   :undoc-members:
   :members: x, y, to

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
