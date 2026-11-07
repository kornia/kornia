Kernels
=======

.. currentmodule:: kornia.filters

Kernel builders. Pass a kernel to a :doc:`filters.filtering_api` function, or build one to see what a filter applies.

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
