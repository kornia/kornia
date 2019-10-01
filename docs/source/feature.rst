kornia.feature
=====================

.. currentmodule:: kornia.feature

.. autofunction:: non_maxima_suppression2d
.. autofunction:: gftt_response
.. autofunction:: harris_response
.. autofunction:: hessian_response
.. autofunction:: extract_patches_from_pyramid
.. autofunction:: extract_patches_simple
.. autofunction:: normalize_laf
.. autofunction:: denormalize_laf
.. autofunction:: laf_to_boundary_points
.. autofunction:: ellipse_to_laf
.. autofunction:: make_upright
.. autofunction:: scale_laf
.. autofunction:: get_laf_scale
.. autofunction:: raise_error_if_laf_is_not_valid

.. autoclass:: NonMaximaSuppression2d
.. autoclass:: BlobHessian
.. autoclass:: CornerGFTT
.. autoclass:: CornerHarris
.. autoclass:: SIFTDescriptor
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
