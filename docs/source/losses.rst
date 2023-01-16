kornia.losses
=============

.. currentmodule:: kornia.losses

Reconstruction
--------------

.. autofunction:: ssim_loss
.. autofunction:: ssim3d_loss
.. autofunction:: psnr_loss
.. autofunction:: total_variation
.. autofunction:: inverse_depth_smoothness_loss

.. autoclass:: SSIMLoss
.. autoclass:: SSIM3DLoss
.. autoclass:: MS_SSIMLoss
.. autoclass:: TotalVariation
.. autoclass:: PSNRLoss
.. autoclass:: InverseDepthSmoothnessLoss

Semantic Segmentation
---------------------

.. autofunction:: binary_focal_loss_with_logits
.. autofunction:: focal_loss
.. autofunction:: dice_loss
.. autofunction:: tversky_loss
.. autofunction:: lovasz_hinge_loss
.. autofunction:: lovasz_softmax_loss

.. autoclass:: BinaryFocalLossWithLogits
.. autoclass:: DiceLoss
.. autoclass:: TverskyLoss
.. autoclass:: FocalLoss
.. autoclass:: LovaszHingeLoss
.. autoclass:: LovaszSoftmaxLoss

Distributions
-------------

.. autofunction:: js_div_loss_2d
.. autofunction:: kl_div_loss_2d

Morphology
----------

.. autoclass:: HausdorffERLoss
.. autoclass:: HausdorffERLoss3D
