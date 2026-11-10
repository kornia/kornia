kornia.metrics
==============

.. meta::
   :description: kornia.metrics scores predictions for classification, segmentation, detection, image quality (PSNR, SSIM), optical flow (AEPE), stereo disparity and pose, and averages them over training batches.

.. currentmodule:: kornia.metrics

Metrics to monitor and evaluate vision models during training and validation.

.. list-table::
   :widths: 30 70

   * - :doc:`metrics.classification`
     - Top-k accuracy.
   * - :doc:`metrics.segmentation`
     - Confusion matrix and mean IoU.
   * - :doc:`metrics.detection`
     - Mean average precision and box IoU.
   * - :doc:`metrics.image_quality`
     - PSNR and SSIM in 2D and 3D.
   * - :doc:`metrics.optical_flow`
     - Average end-point error.
   * - :doc:`metrics.stereo`
     - Disparity error metrics.
   * - :doc:`metrics.pose`
     - Rotation and translation errors and AUC.
   * - :doc:`metrics.monitoring`
     - Running averages for training loops.

.. toctree::
   :hidden:

   classification <metrics.classification>
   segmentation & iou <metrics.segmentation>
   detection <metrics.detection>
   image quality <metrics.image_quality>
   optical flow <metrics.optical_flow>
   stereo <metrics.stereo>
   pose <metrics.pose>
   monitoring <metrics.monitoring>
