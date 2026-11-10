kornia.feature
==============

.. meta::
   :description: The kornia.feature module offers tools to detect, describe, and match local features in images. It includes classical and learned detectors, descriptors and matchers such as Harris, SIFT, KeyNet, DISK, ALIKED, HardNet, LightGlue and LoFTR, and tools for working with local affine frames (LAFs).

.. currentmodule:: kornia.feature

Local feature detection, description and matching, from classical operators to pretrained models, as PyTorch
functions and classes. For guidance on which model to pick, see the
:doc:`image matching guide </applications/image_matching>`.
For classical SIFT, start with the explicit scale-space pyramid recipe in the
:ref:`SIFT pipeline guide <sift-pipelines>`. For supplied patches or frames, see
the :ref:`SIFT descriptor interfaces <sift-descriptor-interfaces>`.

.. list-table::
   :widths: 30 70

   * - :doc:`feature.detectors`
     - Detect LAFs only: the scale-space, multi-resolution and KeyNet detectors, and the Harris, GFTT, Hessian, DoG
       and KeyNet response maps they take.
   * - :doc:`feature.descriptors`
     - Describe patches (SIFT, MKD, HardNet, HardNet8, HyNet, SOSNet, TFeat), the LAFs of an image (LAFDescriptor,
       get_laf_descriptors, SIFT from a pyramid) or every position (dense SIFT); SOLD2 for line segments.
   * - :doc:`feature.local_features`
     - Detect and describe in one call: the learned ALIKED, DISK, XFeat and DeDoDe, the SIFT and HardNet presets of
       LocalFeature, and the SOLD2 line detector.
   * - :doc:`feature.matching`
     - Match descriptors by nearest neighbour, mutual check, ratio test, FGINN or AdaLAM, or with the learned
       LightGlue (LightGlueMatcher, LightGlue, OnnxLightGlue); match two images in one call (LocalFeatureMatcher,
       LoFTR).
   * - :doc:`feature.laf`
     - Local affine frames (LAFs): build, read, transform and check them, sample their patches, and estimate their
       orientation and affine shape.
   * - :doc:`feature.layers`
     - HyNet's FRN and TLU layers, and the DeFMO fast-moving-object model.

.. toctree::
   :hidden:

   feature.detectors
   feature.descriptors
   feature.local_features
   feature.matching
   feature.laf
   feature.layers
