kornia.metrics
==============

.. meta::
   :description: The kornia.metrics module provides a variety of metrics to evaluate the performance of deep learning models in computer vision tasks. It includes metrics for classification, segmentation, detection, image quality, and optical flow. With functions such as accuracy, mean IoU, PSNR, and AEPE, this module facilitates efficient monitoring and evaluation of models during training, making it a valuable tool for model performance assessment.

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
