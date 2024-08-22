Local Features and Image Matching
==============

This module provides a set of tools to detect and describe local features in images. The module is designed to be
compatible with the PyTorch ecosystem and provides a set of models and differentiable operations to be used in deep learning
pipelines.

The module is divided into three main components:

1. **Detectors**: These are models that are used to detect keypoints in images. The module provides a set of detectors that
   are based on different algorithms such as Harris, GFTT, Hessian, and DoG. The module also provides a set of detectors that
   are based on deep learning models such as KeyNet, DISK and DeDoDe.
2. **Descriptors**: These are models that are used to describe the local features detected by the detectors. The module
   provides a set of descriptors that are based on different algorithms such as SIFT, HardNet, and TFeat. The module also
   provides a set of descriptors that are based on deep learning models such as HyNet, SOSNet, and LAFDescriptor.
3. **Matching**: These are models that are used to match the local features detected and described by the detectors and
   descriptors. The module provides a set of matching algorithms such as nearest neighbor, mutual nearest neighbor, and
   geometrically aware matching. Besides this, the module also contains AdaLAM hancrafted and LightGlue learned matchers.
   Finally, the module provides LoFTR - detector-less semi-dense image matching model.

Besides this, the module also provides a set of tools to work with local affine frames (LAF) such as extracting patches,
normalizing, denormalizing, and rotating LAFs. The module also provides a set of models to estimate the affine shape of
LAFs such as LAFAffineShapeEstimator and PatchAffineShapeEstimator. The module also provides a set of models to estimate
the orientation of LAFs such as OriNet and LAFOrienter.


Finally, kind of addition, module contains a DeFMO model for the task of video frame interpolation, specifically high speed objects debluring.

Benchmarks and recommendations
---------

The following table shows the performance of the different models on `IMC2021 benchmark <https://www.cs.ubc.ca/research/image-matching-challenge/2021/leaderboard/>`_ .


.. list-table:: IMC2021 Benchmark, 8000 features
   :widths: 50 50 50 50 50
   :header-rows: 1
   * - Feature name
     - Stereo mAA @ 10 degrees, PhotoTourism.
     - Multiview mAA @ 10 degrees, PhotoTourism.
     - Stereo mAA @ 10 degrees, PragueParks.
     - Multiview mAA @ 10 degrees, PragueParks.
   * - OpenCV-DoG-HardNet-LightGlue
     - 0.5850
     - 0.7587
     - 0.6525
     - 0.4973
   * - OpenCV-DoG-AffNet-HardNet8-AdaLAM
     - 0.5502
     - 0.7522
     - 0.5998
     - 0.4712
   * - DISK-LightGlue
     - 0.6184
     - 0.7741
     - 0.6116
     - 0.4988
   * - LoFTR
     - 0.6090
     - 0.7609
     - 0.7546
     - 0.4711
   * - Upright SIFT (OpenCV)
     - 0.5122
     - 0.6849
     - 0.6060
     - 0.4439


.. list-table:: IMC2021 Benchmark, 2048 features
   :widths: 50 50 50 50 50
   :header-rows: 1
   * - Feature name
     - Stereo mAA @ 10 degrees, PhotoTourism.
     - Multiview mAA @ 10 degrees, PhotoTourism.
     - Stereo mAA @ 10 degrees, PragueParks.
     - Multiview mAA @ 10 degrees, PragueParks.
   * - OpenCV-DoG-HardNet-LightGlue
     - 0.3954
     - 0.6272
     - 0.5157
     - 0.4456
   * - DISK-LightGlue
     - 0.5720
     - 0.7543
     - 0.5099
     - 0.4565
   * - Upright SIFT (OpenCV)
     - 0.3827
     - 0.5545
     - 0.4136
     - 0.3607

LoFTR works the best for indoor scenes, whereas DISK and DeDoDe + LightGlue work the best for outdoor scenes.
The DeDoDe and speed benchmarks are coming soon.


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

.. autoclass:: DeDoDe
   :members: forward, from_pretrained, describe, detect

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

Interactive Demo
~~~~~~~~~~~~~~~~
.. raw:: html

    <gradio-app src="https://kornia-kornia-loftr.hf.space"></gradio-app>

.. autoclass:: OnnxLightGlue
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
