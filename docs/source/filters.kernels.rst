Kernels
=======

.. currentmodule:: kornia.filters

Kernel builders, to see what a filter applies or to filter with a :doc:`filters.filtering_api` function.
:func:`filter2d`, its 2D variants and :func:`fft_conv` take a ``(1, kH, kW)`` kernel: add a leading axis to a
``(kH, kW)`` one, and pass a stacked derivative kernel such as :func:`get_sobel_kernel2d` one kernel at a time, since a
``(2, 3, 3)`` stack is read as one kernel per sample. :func:`filter2d_separable` takes two ``(1, k)`` kernels and
:func:`filter3d` a ``(1, kD, kH, kW)`` one, the shapes the 1D and 3D builders return.

Which Gaussian to use:

- :func:`get_gaussian_kernel1d` samples the Gaussian at the taps. It is the kernel :func:`gaussian_blur2d` applies,
  and :func:`get_gaussian_kernel2d` and :func:`get_gaussian_kernel3d` are its outer products.
- :func:`get_gaussian_erf_kernel1d` integrates the Gaussian over each pixel, for a tap that models a pixel's
  area. It blurs more.
- :func:`get_gaussian_discrete_kernel1d` is Lindeberg's discrete Gaussian, for discrete scale space.
- :func:`gaussian` samples like :func:`get_gaussian_kernel1d`, takes a ``mean`` and does not check the size.

.. autofunction:: gaussian
.. autofunction:: get_gaussian_kernel1d
.. autofunction:: get_gaussian_erf_kernel1d
.. autofunction:: get_gaussian_discrete_kernel1d
.. autofunction:: get_gaussian_kernel2d
.. autofunction:: get_gaussian_kernel3d
.. autofunction:: get_box_kernel1d
.. autofunction:: get_box_kernel2d
.. autofunction:: get_binary_kernel2d
.. autofunction:: get_hanning_kernel1d
.. autofunction:: get_hanning_kernel2d
.. autofunction:: get_laplacian_kernel1d
.. autofunction:: get_laplacian_kernel2d
.. autofunction:: laplacian_1d
.. autofunction:: get_spatial_gradient_kernel2d
.. autofunction:: get_spatial_gradient_kernel3d
.. autofunction:: get_sobel_kernel2d
.. autofunction:: get_diff_kernel2d
.. autofunction:: get_motion_kernel2d
.. autofunction:: get_motion_kernel3d
