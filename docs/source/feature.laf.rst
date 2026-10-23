Local affine frames (LAF)
=========================

.. currentmodule:: kornia.feature

A local affine frame is a :math:`(B, N, 2, 3)` tensor describing, for each keypoint, an affine transformation
from a canonical patch to the image. Most detectors and descriptors in :mod:`kornia.feature` exchange LAFs.

Functions
---------

.. autofunction:: extract_patches_from_pyramid
.. autofunction:: extract_patches_simple
.. autofunction:: normalize_laf
.. autofunction:: denormalize_laf
.. autofunction:: laf_to_boundary_points
.. autofunction:: ellipse_to_laf
.. autofunction:: make_upright
.. autofunction:: scale_laf
.. autofunction:: get_laf_scale
.. autofunction:: get_laf_center
.. autofunction:: rotate_laf
.. autofunction:: get_laf_orientation
.. autofunction:: set_laf_orientation
.. autofunction:: laf_from_center_scale_ori
.. autofunction:: laf_is_filled
.. autofunction:: laf_is_inside_image
.. autofunction:: laf_is_valid
.. autofunction:: laf_to_three_points
.. autofunction:: laf_from_three_points
.. autofunction:: KORNIA_CHECK_LAF
.. autofunction:: perspective_transform_lafs

Orientation and affine shape estimation
---------------------------------------

.. autoclass:: PassLAF
   :members: forward

.. autoclass:: PatchAffineShapeEstimator
   :members: forward

.. autoclass:: LAFAffineShapeEstimator
   :members: forward

.. autoclass:: LAFOrienter
   :members: forward

.. autoclass:: PatchDominantGradientOrientation
   :members: forward

.. autoclass:: OriNet
   :members: forward

.. autoclass:: LAFAffNetShapeEstimator
   :members: forward
