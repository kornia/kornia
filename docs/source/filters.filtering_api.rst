Filtering API
=============

.. currentmodule:: kornia.filters

Filter an image with your own kernel, by correlation (the default) or convolution. These are the primitives the blur
and edge operators are built on.

.. autofunction:: filter2d
.. autofunction:: correlate2d
.. autofunction:: convolve2d
.. autofunction:: filter2d_separable
.. autofunction:: correlate3d
.. autofunction:: convolve3d
.. autofunction:: filter3d
.. autofunction:: fft_conv
