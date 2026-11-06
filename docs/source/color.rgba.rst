RGBA
====

.. currentmodule:: kornia.color

:func:`rgb_to_rgba` appends an alpha channel. :func:`rgba_to_rgb` composites the alpha over a background, white by
default; :func:`rgba_to_bgr` composites over white only.

.. tip::
    Learn more: https://en.wikipedia.org/wiki/RGBA_color_model

Functions
---------

.. autofunction:: rgb_to_rgba
.. autofunction:: rgba_to_rgb
.. autofunction:: rgba_to_bgr

Modules
-------

.. autoclass:: RgbToRgba
.. autoclass:: RgbaToRgb
.. autoclass:: RgbaToBgr
