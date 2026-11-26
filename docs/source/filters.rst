kornia.filters
==============

.. meta::
   :description: The Kornia filters module blurs, sharpens and thresholds images, finds edges and gradients, and filters with your own kernels. It includes box, Gaussian, median, bilateral, guided and motion blurs, unsharp masking, blur pooling, Sobel, Laplacian and Canny edges, and Otsu thresholding.

.. currentmodule:: kornia.filters

Filters for images ``(B, C, H, W)`` and volumes ``(B, C, D, H, W)``. Each blurring, edge detection and thresholding
function has an ``nn.Module`` form; the filtering API and the kernel functions are functions only.

.. list-table::
   :widths: 30 70

   * - :doc:`filters.blurring`
     - Gaussian, box, median, bilateral, guided and motion blur; blur pooling; unsharp masking; dissolving with
       Stable Diffusion.
   * - :doc:`filters.edge_detection`
     - Sobel, Laplacian and Canny edge detectors and spatial gradients in 2D and 3D.
   * - :doc:`Thresholding <filters.segmentation>`
     - Range thresholding and Otsu's threshold.
   * - :doc:`filters.filtering_api`
     - Apply your own 2D, separable or 3D kernels with :func:`filter2d`, :func:`filter2d_separable` and :func:`filter3d`,
       or by correlation or convolution with :func:`correlate2d`, :func:`convolve2d`, :func:`correlate3d` and
       :func:`convolve3d`; :func:`fft_conv` filters in the frequency domain.
   * - :doc:`filters.kernels`
     - Gaussian, box, Hanning, Laplacian, gradient and motion kernels used by the filters above.

.. toctree::
   :hidden:

   blurring <filters.blurring>
   edge detection <filters.edge_detection>
   thresholding <filters.segmentation>
   filtering api <filters.filtering_api>
   kernels <filters.kernels>
