kornia.losses
=============

.. meta::
   :name: description
   :content: "The kornia.losses module offers a comprehensive collection of loss functions for computer vision tasks, including image reconstruction, semantic segmentation, distribution-based losses, and morphological losses. With a wide range of loss types such as SSIM, PSNR, focal loss, and dice loss, this module enables efficient optimization for deep learning models across various domains, enhancing training for tasks like image restoration, segmentation, and object detection."

.. currentmodule:: kornia.losses

Reconstruction
--------------

.. autofunction:: ssim_loss
.. autofunction:: ssim3d_loss
.. autofunction:: psnr_loss
.. autofunction:: total_variation
.. autofunction:: inverse_depth_smoothness_loss
.. autofunction:: charbonnier_loss
.. autofunction:: welsch_loss
.. autofunction:: cauchy_loss
.. autofunction:: geman_mcclure_loss

.. autoclass:: SSIMLoss
.. autoclass:: SSIM3DLoss
.. autoclass:: MS_SSIMLoss
.. autoclass:: TotalVariation
.. autoclass:: PSNRLoss
.. autoclass:: InverseDepthSmoothnessLoss
.. autoclass:: CharbonnierLoss
.. autoclass:: WelschLoss
.. autoclass:: CauchyLoss
.. autoclass:: GemanMcclureLoss

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
