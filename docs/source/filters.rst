kornia.filters
==============

.. currentmodule:: kornia.filters

The functions in this sections perform various image filtering operations.

Blurring
--------

.. autofunction:: blur_pool2d
.. autofunction:: box_blur
.. autofunction:: gaussian_blur2d
.. autofunction:: max_blur_pool2d
.. autofunction:: median_blur
.. autofunction:: motion_blur
.. autofunction:: unsharp_mask

Edge detection
--------------

.. autofunction:: canny
.. autofunction:: laplacian
.. autofunction:: sobel
.. autofunction:: spatial_gradient
.. autofunction:: spatial_gradient3d

Filtering API
-------------

.. autofunction:: filter2d
.. autofunction:: filter2d_separable
.. autofunction:: filter3d

Kernels
-------

.. autofunction:: get_gaussian_kernel1d
.. autofunction:: get_gaussian_erf_kernel1d
.. autofunction:: get_gaussian_discrete_kernel1d
.. autofunction:: get_gaussian_kernel2d
.. autofunction:: get_hanning_kernel1d
.. autofunction:: get_hanning_kernel2d
.. autofunction:: get_laplacian_kernel1d
.. autofunction:: get_laplacian_kernel2d
.. autofunction:: get_motion_kernel2d

Module
------

.. autoclass:: BlurPool2D
.. autoclass:: BoxBlur
.. autoclass:: MaxBlurPool2D
.. autoclass:: MedianBlur
.. autoclass:: GaussianBlur2d
.. autoclass:: Laplacian
.. autoclass:: Sobel
.. autoclass:: Canny
.. autoclass:: SpatialGradient
.. autoclass:: SpatialGradient3d
.. autoclass:: MotionBlur
.. autoclass:: UnsharpMask
