kornia.losses
=============

.. meta::
   :description: The kornia.losses module offers a comprehensive collection of loss functions for computer vision tasks, including image reconstruction, semantic segmentation, distribution-based losses, and morphological losses. With a wide range of loss types such as SSIM, PSNR, focal loss, and dice loss, this module enables efficient optimization for deep learning models across various domains, enhancing training for tasks like image restoration, segmentation, and object detection.

.. currentmodule:: kornia.losses

Loss functions for training vision models. Each functional loss has an ``nn.Module`` counterpart.

.. list-table::
   :widths: 30 70

   * - :doc:`losses.reconstruction`
     - SSIM, MS-SSIM, PSNR, total variation, inverse-depth smoothness and robust losses (Charbonnier, Welsch, Cauchy, Geman-McClure).
   * - :doc:`losses.segmentation`
     - Focal, dice, Tversky and Lovasz losses.
   * - :doc:`losses.distributions`
     - Jensen-Shannon and Kullback-Leibler divergences between 2D distributions.
   * - :doc:`losses.morphology`
     - Hausdorff distance losses in 2D and 3D.

.. toctree::
   :hidden:

   reconstruction <losses.reconstruction>
   segmentation <losses.segmentation>
   distributions <losses.distributions>
   morphology <losses.morphology>
