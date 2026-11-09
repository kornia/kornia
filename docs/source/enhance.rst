kornia.enhance
==============

.. meta::
   :description: The Kornia.enhance module provides a suite of image enhancement functions including brightness, contrast, hue, saturation adjustments, as well as normalization and equalization techniques. It also features advanced transformations like ZCA whitening and differentiable JPEG codec.

.. currentmodule:: kornia.enhance

Adjust, equalize, normalize and whiten images, and simulate JPEG compression.

.. list-table::
   :widths: 30 70

   * - :doc:`enhance.adjustment`
     - Brightness, contrast, gamma, logarithmic, sigmoid, hue and saturation adjustments; invert, posterize, solarize, sharpen, RGB shift, weighted sum, rescale and fixed-level threshold.
   * - :doc:`enhance.equalization`
     - Histogram equalization, tile-wise CLAHE and differentiable histograms.
   * - :doc:`enhance.normalization`
     - Mean and standard-deviation normalization, min-max scaling, ZCA whitening and linear transforms.
   * - :doc:`enhance.codec`
     - A differentiable JPEG compression and decompression model.
   * - :doc:`enhance.integral`
     - Integral images and cumulative sums over chosen axes.

.. toctree::
   :hidden:

   adjustment <enhance.adjustment>
   equalization <enhance.equalization>
   normalization <enhance.normalization>
   codec <enhance.codec>
   integral <enhance.integral>
