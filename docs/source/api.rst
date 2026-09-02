API reference
=============

.. meta::
   :description: Reference documentation for every public Kornia module: augmentation, color, contrib, core, enhance, feature, filters, geometry, sensors, io, image, losses, models, metrics, morphology, onnx and tracking.

One page per ``kornia`` module, grouped by what you are trying to do. Image operators take batched
``(B, C, H, W)`` float tensors in ``[0, 1]`` and follow the :doc:`conventions </get-started/conventions>`
(points, boxes, local features and camera matrices have their own layouts, documented on their pages);
most functions also have an ``nn.Module`` counterpart.

Image processing
----------------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.color <color>`
     - Color space conversions, color maps and Bayer RAW processing.
   * - :doc:`kornia.filters <filters>`
     - Blurring, edge detection, thresholding and custom kernels.
   * - :doc:`kornia.enhance <enhance>`
     - Intensity adjustments, histogram equalization, normalization and a differentiable JPEG codec.
   * - :doc:`kornia.morphology <morphology>`
     - Dilation, erosion, opening, closing, gradient, top hat and bottom hat.

Geometry & 3D
-------------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.geometry <geometry>`
     - Warps, camera models, conversions, epipolar geometry, Lie groups, RANSAC and more.
   * - :doc:`kornia.sensors <sensors>`
     - Experimental camera model API.

Training
--------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.augmentation <augmentation>`
     - Random and deterministic augmentations for images, masks, boxes, keypoints and video, with transform tracking and inversion.
   * - :doc:`kornia.losses <losses>`
     - Reconstruction, segmentation, distribution and morphology losses.
   * - :doc:`kornia.metrics <metrics>`
     - Classification, segmentation, detection, image quality, flow, stereo and pose metrics.

Features & matching
-------------------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.feature <feature>`
     - Local feature detectors, descriptors and matchers, classical and learned.

Models
------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.models <models>`
     - Builders for the pretrained detection, edge detection, segmentation and tracking models.
   * - :doc:`kornia.contrib <contrib>`
     - Experimental operators and model wrappers: face detection, object detection, visual prompting, image stitching, KMeans.
   * - :doc:`kornia.tracking <tracking>`
     - Homography tracking.

Data & deployment
-----------------

.. list-table::
   :widths: 28 72

   * - :doc:`kornia.io <io>`
     - Load and save images as tensors.
   * - :doc:`kornia.image <image>`
     - Image container, drawing, tensor/NumPy conversion.
   * - :doc:`kornia.onnx <onnx>`
     - Run and chain ONNX models with ONNX Runtime.
   * - :doc:`kornia.core <core>`
     - Tensor wrapper and shared utilities.

.. toctree::
   :caption: Image processing
   :hidden:

   kornia.color — color spaces <color>
   kornia.filters — blur & edges <filters>
   kornia.enhance — intensity & histograms <enhance>
   kornia.morphology — dilate & erode <morphology>

.. toctree::
   :caption: Geometry & 3D
   :hidden:

   kornia.geometry — warps, cameras, 3D <geometry>
   kornia.sensors — camera models <sensors>

.. toctree::
   :caption: Training
   :hidden:

   kornia.augmentation — transforms & policies <augmentation>
   kornia.losses — training objectives <losses>
   kornia.metrics — evaluation <metrics>

.. toctree::
   :caption: Features & matching
   :hidden:

   kornia.feature — detect & match <feature>

.. toctree::
   :caption: Models
   :hidden:

   kornia.models — pretrained builders <models>
   kornia.contrib — experimental <contrib>
   kornia.tracking — homography tracking <tracking>

.. toctree::
   :caption: Data & deployment
   :hidden:

   kornia.io — read & write images <io>
   kornia.image — image container <image>
   kornia.onnx — ONNX runtime <onnx>
   kornia.core — tensor utilities <core>
