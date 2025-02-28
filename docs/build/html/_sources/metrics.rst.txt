kornia.metrics
==============

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
