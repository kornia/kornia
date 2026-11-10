Local affine frames (LAF)
=========================

.. currentmodule:: kornia.feature

A local affine frame (LAF) describes a keypoint by a ``2 x 3`` matrix ``[A | c]``, and a set of them is a
``(B, N, 2, 3)`` tensor. The centre ``c = (x, y)`` is in pixels, and the shape ``A`` maps the canonical patch frame
onto the image, in pixels: its scale is :math:`\sqrt{|\det A|}` and its orientation ``atan2(a12, a11)``, in degrees
counter-clockwise on screen for a LAF of positive determinant (clockwise for a mirrored one).
:ref:`Local features <local-features-conventions>` gives the frame and the conversion of OpenCV keypoints. kornia's
scale-space and KeyNet detectors (:class:`ScaleSpaceDetector`, :class:`MultiResolutionDetector`,
:class:`KeyNetDetector`) return LAFs, :class:`LAFDescriptor` describes them, and :func:`match_fginn`,
:func:`match_adalam` and :class:`LightGlueMatcher` read them; :func:`laf_from_center_scale_ori` turns the keypoints of
the other detectors into LAFs.

- To keep the detections of a fixed-size detector output, use :func:`laf_is_filled`; to also drop singular and
  non-finite frames, :func:`laf_is_valid`; to keep the regions that lie inside the image, :func:`laf_is_inside_image`.
- To sample patches to describe, use :func:`extract_patches_from_pyramid`, which reads each LAF from a pyramid level
  chosen by its scale and limits aliasing. :func:`extract_patches_simple` reads the full-resolution image only and
  aliases for large LAFs; a highly anisotropic LAF can still alias along its long axis.
- To drop the orientation, use :func:`make_upright`; to set or add angles, :func:`set_laf_orientation` or
  :func:`rotate_laf`; to estimate the orientation from the image, :class:`LAFOrienter`.
- To adapt the affine shape to the image, use :class:`LAFAffineShapeEstimator` (the second-moment matrix of the
  patch) or :class:`LAFAffNetShapeEstimator` (the learned AffNet); :class:`PassLAF` leaves the LAFs as they are.

Creating and reading LAFs
-------------------------

.. autofunction:: laf_from_center_scale_ori
.. autofunction:: ellipse_to_laf
.. autofunction:: laf_from_three_points
.. autofunction:: get_laf_center
.. autofunction:: get_laf_scale
.. autofunction:: get_laf_orientation
.. autofunction:: laf_to_three_points
.. autofunction:: laf_to_boundary_points

Transforming LAFs
-----------------

.. autofunction:: scale_laf
.. autofunction:: rotate_laf
.. autofunction:: set_laf_orientation
.. autofunction:: make_upright
.. autofunction:: perspective_transform_lafs
.. autofunction:: normalize_laf
.. autofunction:: denormalize_laf

Checking LAFs
-------------

.. autofunction:: laf_is_valid
.. autofunction:: laf_is_filled
.. autofunction:: laf_is_inside_image
.. autofunction:: KORNIA_CHECK_LAF

Extracting patches
------------------

.. autofunction:: extract_patches_from_pyramid
.. autofunction:: extract_patches_simple

Orientation and affine shape estimation
---------------------------------------

.. autoclass:: LAFOrienter
   :members: forward

.. autoclass:: PatchDominantGradientOrientation
   :members: forward

.. autoclass:: OriNet
   :members: forward

.. autoclass:: LAFAffineShapeEstimator
   :members: forward

.. autoclass:: PatchAffineShapeEstimator
   :members: forward

.. autoclass:: LAFAffNetShapeEstimator
   :members: forward

.. autoclass:: PassLAF
   :members: forward
