kornia.losses
=============

.. currentmodule:: kornia.losses

Reconstruction
--------------

.. autofunction:: ssim_loss
.. autofunction:: psnr_loss
.. autofunction:: total_variation
.. autofunction:: inverse_depth_smoothness_loss

Semantic Segmentation
---------------------

.. autofunction:: binary_focal_loss_with_logits
.. autofunction:: focal_loss
.. autofunction:: dice_loss
.. autofunction:: tversky_loss

Distributions
-------------

.. autofunction:: js_div_loss_2d
.. autofunction:: kl_div_loss_2d

Morphology
----------

.. autoclass:: HausdorffERLoss
.. autoclass:: HausdorffERLoss3D

Module
------

.. autoclass:: DiceLoss
.. autoclass:: TverskyLoss
.. autoclass:: FocalLoss
.. autoclass:: SSIMLoss
.. autoclass:: MS_SSIMLoss
.. autoclass:: InverseDepthSmoothnessLoss
.. autoclass:: TotalVariation
.. autoclass:: PSNRLoss
.. autoclass:: BinaryFocalLossWithLogits
