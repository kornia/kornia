kornia.enhance
==============

.. meta::
   :description: The Kornia.enhance module provides a suite of image enhancement functions including brightness, contrast, hue, saturation adjustments, as well as normalization and equalization techniques. It also features advanced transformations like ZCA whitening and differentiable JPEG codec. Explore interactive demos on Hugging Face Spaces.

.. currentmodule:: kornia.enhance

Intensity transformations and normalization on batched ``(B, C, H, W)`` tensors.

.. list-table::
   :widths: 30 70

   * - :doc:`enhance.adjustment`
     - Brightness, contrast, gamma, hue, saturation, sigmoid and log adjustments; invert, posterize, sharpen and solarize.
   * - :doc:`enhance.equalization`
     - Histogram equalization, CLAHE and differentiable histograms.
   * - :doc:`enhance.normalization`
     - Mean/std normalization, min-max scaling, ZCA whitening and linear transforms.
   * - :doc:`enhance.codec`
     - A differentiable JPEG encoder/decoder.

.. toctree::
   :hidden:

   adjustment <enhance.adjustment>
   equalization <enhance.equalization>
   normalization <enhance.normalization>
   codec <enhance.codec>
