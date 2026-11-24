Distributions
=============

.. currentmodule:: kornia.losses

Both losses compare ``(B, N, H, W)`` probability heatmaps. Use :func:`kl_div_loss_2d` when ``target`` is the reference
distribution, and :func:`js_div_loss_2d` for a symmetric loss that stays finite where one heatmap is zero.

.. autofunction:: js_div_loss_2d
.. autofunction:: kl_div_loss_2d
