CIE XYZ
=======

.. currentmodule:: kornia.color

.. tip::
    Learn more: https://en.wikipedia.org/wiki/CIE_1931_color_space

.. note::
    :func:`rgb_to_xyz` takes linear RGB: convert an sRGB image with :func:`rgb_to_linear_rgb` first, and the result of
    :func:`xyz_to_rgb` back with :func:`linear_rgb_to_rgb`. :func:`rgb_to_lab` and :func:`rgb_to_luv` take sRGB.

Functions
---------

.. autofunction:: rgb_to_xyz
.. autofunction:: xyz_to_rgb

Modules
-------

.. autoclass:: RgbToXyz
.. autoclass:: XyzToRgb
