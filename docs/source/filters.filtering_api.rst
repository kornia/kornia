Filtering API
=============

.. currentmodule:: kornia.filters

Filter an image with your own kernel. The box, Gaussian and motion blurs and :func:`laplacian` are built on
:func:`filter2d`, :func:`filter2d_separable` and :func:`filter3d`; use :func:`gaussian_blur2d` or :func:`box_blur`
for a Gaussian or box kernel.

Which function to use:

- A 2D kernel, correlated (the default) or convolved: :func:`filter2d`, with ``behaviour='corr'`` or ``'conv'``.
  :func:`correlate2d` and :func:`convolve2d` fix ``behaviour``. They give the same output for a kernel that is its
  own 180-degree rotation.
- A kernel that is the outer product of two 1D kernels: :func:`filter2d_separable`.
- A large 2D kernel: :func:`fft_conv`, which returns the result of :func:`filter2d` through FFTs.
- A volume: :func:`filter3d`, :func:`correlate3d` and :func:`convolve3d`.

:ref:`Filtering conventions <filtering-conventions>` give the kernel anchor, the border modes and the scipy and OpenCV
equivalents.

.. autofunction:: filter2d
.. autofunction:: correlate2d
.. autofunction:: convolve2d
.. autofunction:: filter2d_separable
.. autofunction:: correlate3d
.. autofunction:: convolve3d
.. autofunction:: filter3d
.. autofunction:: fft_conv
