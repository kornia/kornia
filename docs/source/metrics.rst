kornia.metrics
==============

.. meta::
   :name: description
   :content: "The kornia.metrics module provides a variety of metrics to evaluate the performance of deep learning models in computer vision tasks. It includes metrics for classification, segmentation, detection, image quality, and optical flow. With functions such as accuracy, mean IoU, PSNR, and AEPE, this module facilitates efficient monitoring and evaluation of models during training, making it a valuable tool for model performance assessment."

.. currentmodule:: kornia.metrics

Module containing metrics for training networks

Classification
--------------

.. autofunction:: accuracy

Segmentation
------------

.. autofunction:: confusion_matrix
.. autofunction:: mean_iou

Detection
---------

.. autofunction:: mean_average_precision
.. autofunction:: mean_iou_bbox

Image Quality
-------------

.. autofunction:: psnr
.. autofunction:: ssim
.. autofunction:: ssim3d
.. autoclass:: SSIM
.. autoclass:: SSIM3D

Optical Flow
-------------

.. autofunction:: aepe
.. autoclass:: AEPE

Monitoring
----------

.. autoclass:: AverageMeter
