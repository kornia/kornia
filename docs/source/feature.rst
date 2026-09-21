kornia.feature
==============

.. meta::
   :description: The kornia.feature module offers tools to detect, describe, and match local features in images. It includes classical and learned detectors, descriptors and matchers such as Harris, SIFT, KeyNet, DISK, ALIKED, HardNet, LightGlue and LoFTR, and tools for working with local affine frames (LAFs).

.. currentmodule:: kornia.feature

Local feature detection, description and matching, from classical operators to pretrained models, all as
differentiable PyTorch modules. For guidance on which model to pick, see the
:doc:`image matching guide </applications/image_matching>`.

.. list-table::
   :widths: 30 70

   * - :doc:`feature.detectors`
     - Keypoint detectors: Harris, GFTT, Hessian, DoG responses and the KeyNet, scale-space and multi-resolution detectors.
   * - :doc:`feature.descriptors`
     - Patch descriptors: SIFT, MKD, HardNet, HardNet8, HyNet, TFeat, SOSNet, and the SOLD2 line descriptor.
   * - :doc:`feature.local_features`
     - Detector + descriptor pipelines: DISK, ALIKED, DeDoDe, XFeat, SIFTFeature, KeyNetAffNetHardNet and friends.
   * - :doc:`feature.matching`
     - Nearest-neighbour, mutual, ratio-test, FGINN and AdaLAM matching; LightGlue and LoFTR.
   * - :doc:`feature.laf`
     - Local affine frames (LAFs): patch extraction, normalization, orientation and affine shape estimation.
   * - :doc:`feature.layers`
     - Building-block layers and the DeFMO fast-moving-object model.

.. toctree::
   :hidden:

   feature.detectors
   feature.descriptors
   feature.local_features
   feature.matching
   feature.laf
   feature.layers
