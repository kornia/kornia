kornia.feature
==============

.. currentmodule:: kornia.feature

Non Maxima Suppression
----------------------

.. autofunction:: non_maxima_suppression2d
.. autofunction:: non_maxima_suppression3d
.. autofunction:: nms2d
.. autofunction:: nms3d

Detectors
---------

.. autofunction:: gftt_response
.. autofunction:: harris_response
.. autofunction:: hessian_response
.. autofunction:: dog_response


Descriptors
-----------

.. autoclass:: SIFTDescriptor
.. autoclass:: HardNet
.. autoclass:: SOSNet


Matching
-----------

.. autofunction:: match_nn
.. autofunction:: match_mnn
.. autofunction:: match_snn
.. autofunction:: match_smnn


Local Affine Frames (LAF)
-------------------------

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
.. autofunction:: get_laf_orientation
.. autofunction:: laf_from_center_scale_ori
.. autofunction:: laf_is_inside_image
.. autofunction:: laf_to_three_points
.. autofunction:: laf_from_three_points
.. autofunction:: raise_error_if_laf_is_not_valid


Module
------

.. autoclass:: NonMaximaSuppression2d
.. autoclass:: NonMaximaSuppression3d
.. autoclass:: BlobHessian
.. autoclass:: CornerGFTT
.. autoclass:: CornerHarris
.. autoclass:: BlobDoG


.. autoclass:: ScaleSpaceDetector
   :members: forward
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
