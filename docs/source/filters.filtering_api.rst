Filtering API
=============

.. currentmodule:: kornia.filters

Filter an image with your own kernel, by correlation (the default) or convolution. The box, Gaussian and motion blurs
and :func:`laplacian` are built on :func:`filter2d`, :func:`filter2d_separable` and :func:`filter3d`.

.. autofunction:: filter2d
.. autofunction:: correlate2d
.. autofunction:: convolve2d
.. autofunction:: filter2d_separable
.. autofunction:: correlate3d
.. autofunction:: convolve3d
.. autofunction:: filter3d
.. autofunction:: fft_conv
