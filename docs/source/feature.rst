kornia.feature
==============

.. currentmodule:: kornia.feature

Detectors
---------

.. autofunction:: gftt_response
.. autofunction:: harris_response
.. autofunction:: hessian_response
.. autofunction:: dog_response
.. autofunction:: dog_response_single
.. autoclass:: BlobHessian
.. autoclass:: CornerGFTT
.. autoclass:: CornerHarris
.. autoclass:: BlobDoG
.. autoclass:: BlobDoGSingle
.. autoclass:: KeyNet
.. autoclass:: MultiResolutionDetector
   :members: forward, remove_borders, detect_features_on_single_level, detect


.. autoclass:: ScaleSpaceDetector
  :members: forward

.. autoclass:: KeyNetDetector
   :members: forward

Descriptors
-----------

.. autoclass:: DenseSIFTDescriptor
.. autoclass:: SIFTDescriptor
.. autoclass:: MKDDescriptor
.. autoclass:: HardNet
.. autoclass:: HardNet8
.. autoclass:: HyNet
.. autoclass:: TFeat
.. autoclass:: SOSNet
.. autoclass:: LAFDescriptor
   :members: forward

.. autoclass:: SOLD2
   :members: forward

.. autofunction:: get_laf_descriptors



Local Features (Detector and Descriptors together)
---------------------------------------------------


.. autoclass:: LocalFeature
   :members: forward

.. autoclass:: SOLD2_detector
   :members: forward

.. autoclass:: DISK
   :members: forward, from_pretrained, heatmap_and_dense_descriptors

.. autoclass:: DISKFeatures
   :undoc-members:
   :members: x, y, to

.. autoclass:: SIFTFeature
   :members: forward

.. autoclass:: SIFTFeatureScaleSpace
   :members: forward

.. autoclass:: GFTTAffNetHardNet
   :members: forward

.. autoclass::HesAffNetHardNet
   :members: forward

.. autoclass:: KeyNetAffNetHardNet
   :members: forward

.. autoclass:: KeyNetHardNet
   :members: forward


Matching
--------

.. autofunction:: match_nn
.. autofunction:: match_mnn
.. autofunction:: match_snn
.. autofunction:: match_smnn
.. autofunction:: match_fginn
.. autofunction:: match_adalam

.. autoclass:: DescriptorMatcher
   :members: forward

.. autoclass:: GeometryAwareDescriptorMatcher
   :members: forward

.. autoclass:: LocalFeatureMatcher
   :members: forward

.. autoclass:: LightGlueMatcher
   :members: forward

.. autoclass:: LightGlue
   :members: forward

.. autoclass:: LoFTR
   :members: forward


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
.. autofunction:: rotate_laf
.. autofunction:: get_laf_orientation
.. autofunction:: set_laf_orientation
.. autofunction:: laf_from_center_scale_ori
.. autofunction:: laf_is_inside_image
.. autofunction:: laf_to_three_points
.. autofunction:: laf_from_three_points
.. autofunction:: KORNIA_CHECK_LAF
.. autofunction:: perspective_transform_lafs

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

Layers
------

.. autoclass:: FilterResponseNorm2d
.. autoclass:: TLU

Other
------

.. autoclass:: DeFMO
   :members: forward
