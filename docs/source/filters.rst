kornia.filters
==============

.. currentmodule:: kornia.filters

The functions in this sections perform various image filtering operations.

Blurring
--------

.. autofunction:: filter2d
.. autofunction:: filter3d
.. autofunction:: box_blur
.. autofunction:: median_blur
.. autofunction:: gaussian_blur2d
.. autofunction:: motion_blur
.. autofunction:: unsharp_mask

Kernels
-------

.. autofunction:: get_gaussian_kernel1d
.. autofunction:: get_gaussian_erf_kernel1d
.. autofunction:: get_gaussian_discrete_kernel1d
.. autofunction:: get_gaussian_kernel2d
.. autofunction:: get_laplacian_kernel1d
.. autofunction:: get_laplacian_kernel2d
.. autofunction:: get_motion_kernel2d


Edge detection
--------------

.. autofunction:: laplacian
.. autofunction:: sobel
.. autofunction:: canny
.. autofunction:: spatial_gradient
.. autofunction:: spatial_gradient3d

Module
------

.. autoclass:: BoxBlur
.. autoclass:: MedianBlur
.. autoclass:: GaussianBlur2d
.. autoclass:: Laplacian
.. autoclass:: Sobel
.. autoclass:: Canny
.. autoclass:: SpatialGradient
.. autoclass:: SpatialGradient3d
.. autoclass:: MotionBlur
.. autoclass:: UnsharpMask
