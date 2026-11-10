RGB
===

.. currentmodule:: kornia.color

:func:`rgb_to_bgr` reverses the channels. :func:`rgb_to_rgb255` and :func:`rgb255_to_rgb` scale between
:math:`[0, 1]` and :math:`[0, 255]`. :func:`normals_to_rgb255` and :func:`rgb255_to_normals` map surface normals between
:math:`[-1, 1]` and :math:`[0, 255]` colours.

.. tip::
    Learn more: https://en.wikipedia.org/wiki/RGB_color_model

Functions
---------

.. autofunction:: rgb_to_bgr
.. autofunction:: rgb_to_rgb255
.. autofunction:: rgb255_to_rgb
.. autofunction:: rgb255_to_normals
.. autofunction:: normals_to_rgb255

Modules
-------

.. autoclass:: RgbToBgr
.. autoclass:: RgbToRgb255
.. autoclass:: Rgb255ToRgb
.. autoclass:: Rgb255ToNormals
.. autoclass:: NormalsToRgb255
