kornia.filters
==============

.. meta::
   :description: The Kornia filters module provides various image filtering operations such as blurring, edge detection, and noise reduction. It includes functions for bilateral, Gaussian, motion, median, and unsharp mask filtering, as well as pooling operations for blurring. These operations are designed to be differentiable and can be integrated seamlessly into deep learning pipelines.

.. currentmodule:: kornia.filters

Differentiable image filtering on batched ``(B, C, H, W)`` tensors. Every function has an ``nn.Module``
counterpart on the same page.

.. list-table::
   :widths: 30 70

   * - :doc:`filters.blurring`
     - Gaussian, box, median, bilateral, guided and motion blur; blur pooling; unsharp masking.
   * - :doc:`filters.edge_detection`
     - Sobel, Laplacian and Canny edge detectors and spatial gradients in 2D and 3D.
   * - :doc:`Thresholding <filters.segmentation>`
     - Range thresholding and Otsu's threshold.
   * - :doc:`filters.filtering_api`
     - Apply your own 2D, separable or 3D kernels with :func:`filter2d`, :func:`filter2d_separable` and :func:`filter3d`.
   * - :doc:`filters.kernels`
     - Gaussian, Hanning, Laplacian and motion kernels used by the filters above.

.. toctree::
   :hidden:

   blurring <filters.blurring>
   edge detection <filters.edge_detection>
   thresholding <filters.segmentation>
   filtering api <filters.filtering_api>
   kernels <filters.kernels>
