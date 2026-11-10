kornia.losses
=============

.. meta::
   :description: The kornia.losses module offers differentiable losses for image reconstruction, robust regression, semantic segmentation, probability heatmaps, morphology and image registration, such as SSIM, PSNR, focal, dice, Hausdorff and mutual-information losses.

.. currentmodule:: kornia.losses

Score images, segmentations and distributions, or register images by mutual information. A loss with a target takes the
prediction first; :ref:`Losses and metrics <losses-metrics-conventions>` tabulates the ``reduction``
defaults. Each loss function except :func:`kl_div_loss_2d` and :func:`js_div_loss_2d` has a module form.

.. list-table::
   :widths: 30 70

   * - :doc:`losses.reconstruction`
     - SSIM, MS-SSIM and PSNR losses, total variation, inverse-depth smoothness and the robust losses (Charbonnier, Cauchy, Geman-McClure, Welsch).
   * - :doc:`losses.segmentation`
     - Dice, Tversky, Lovász and focal losses on class logits, and :func:`one_hot`.
   * - :doc:`losses.distributions`
     - Jensen-Shannon and Kullback-Leibler divergences between probability heatmaps.
   * - :doc:`losses.morphology`
     - Erosion-based Hausdorff losses for images and volumes.
   * - :doc:`losses.registration`
     - Mutual information and normalized mutual information losses (flat signals, 2D and 3D) for image registration.

.. toctree::
   :hidden:

   reconstruction <losses.reconstruction>
   segmentation <losses.segmentation>
   distributions <losses.distributions>
   morphology <losses.morphology>
   registration <losses.registration>
