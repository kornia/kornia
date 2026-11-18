Grayscale
=========

.. currentmodule:: kornia.color

Grayscale images have one channel, ``(*, 1, H, W)``. :func:`rgb_to_grayscale` and :func:`bgr_to_grayscale` use ITU-R
BT.601 luma weights on the values as given, without removing the sRGB curve; :func:`grayscale_to_rgb` repeats the
channel.

.. tip::
    Learn more: https://en.wikipedia.org/wiki/Grayscale

Functions
---------

.. autofunction:: rgb_to_grayscale
.. autofunction:: bgr_to_grayscale
.. autofunction:: grayscale_to_rgb

Modules
-------

.. autoclass:: RgbToGrayscale
.. autoclass:: BgrToGrayscale
.. autoclass:: GrayscaleToRgb
