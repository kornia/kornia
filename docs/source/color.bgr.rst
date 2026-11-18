BGR
===

.. currentmodule:: kornia.color

BGR is the channel order OpenCV uses. :func:`bgr_to_rgb` reverses the channels; :func:`bgr_to_rgba` also appends an
alpha channel and returns RGBA, not BGRA.

.. tip::
    Learn more: https://en.wikipedia.org/wiki/RGB_color_model

Functions
---------

.. autofunction:: bgr_to_rgb
.. autofunction:: bgr_to_rgba

Modules
-------

.. autoclass:: BgrToRgb
.. autoclass:: BgrToRgba
