Reconstruction
==============

.. currentmodule:: kornia.losses

Compare an image with a reference using :func:`ssim_loss`, :class:`MS_SSIMLoss` or :func:`psnr_loss`. A robust loss
compares any two tensors: :func:`charbonnier_loss` grows linearly for large residuals, :func:`cauchy_loss` only
logarithmically, and :func:`geman_mcclure_loss` and :func:`welsch_loss` saturate. :func:`total_variation` and
:func:`inverse_depth_smoothness_loss` penalise one image or depth map, with no reference.

Functions
---------

.. autofunction:: ssim_loss
.. autofunction:: ssim3d_loss
.. autofunction:: psnr_loss
.. autofunction:: total_variation
.. autofunction:: inverse_depth_smoothness_loss
.. autofunction:: charbonnier_loss
.. autofunction:: welsch_loss
.. autofunction:: cauchy_loss
.. autofunction:: geman_mcclure_loss

Modules
-------

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
