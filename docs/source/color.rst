kornia.color
============

.. currentmodule:: kornia.color

The functions in this section perform various color space conversions.

.. note::
   Check a tutorial for color space conversions `here <https://kornia-tutorials.readthedocs.io/en/latest/
   hello_world_tutorial.html>`__.


Grayscale
---------

.. tip::
    Learn more: https://en.wikipedia.org/wiki/Grayscale

.. autofunction:: rgb_to_grayscale
.. autofunction:: bgr_to_grayscale
.. autofunction:: grayscale_to_rgb

.. autoclass:: GrayscaleToRgb
.. autoclass:: RgbToGrayscale
.. autoclass:: BgrToGrayscale

RGB
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/RGB_color_model

.. autofunction:: rgb_to_bgr
.. autofunction:: bgr_to_rgb

.. autofunction:: rgb_to_linear_rgb
.. autofunction:: linear_rgb_to_rgb

.. autoclass:: RgbToBgr
.. autoclass:: BgrToRgb

.. autoclass:: LinearRgbToRgb
.. autoclass:: RgbToLinearRgb

RGBA
----

.. tip::
    Learn more: https://en.wikipedia.org/wiki/RGBA_color_model

.. autofunction:: bgr_to_rgba
.. autofunction:: rgb_to_rgba
.. autofunction:: rgba_to_rgb
.. autofunction:: rgba_to_bgr

.. autoclass:: RgbToRgba
.. autoclass:: BgrToRgba
.. autoclass:: RgbaToRgb
.. autoclass:: RgbaToBgr

HLS
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/HSL_and_HSV

.. autofunction:: rgb_to_hls
.. autofunction:: hls_to_rgb

.. autoclass:: RgbToHls
.. autoclass:: HlsToRgb

HSV
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/HSL_and_HSV

.. autofunction:: rgb_to_hsv
.. autofunction:: hsv_to_rgb

.. autoclass:: RgbToHsv
.. autoclass:: HsvToRgb

LUV
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/CIELUV

.. autofunction:: rgb_to_luv
.. autofunction:: luv_to_rgb

.. autoclass:: RgbToLuv
.. autoclass:: LuvToRgb

Lab
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/CIELAB_color_space

.. autofunction:: rgb_to_lab
.. autofunction:: lab_to_rgb

.. autoclass:: RgbToLab
.. autoclass:: LabToRgb

YCbCr
-----

.. tip::
    Learn more: https://en.wikipedia.org/wiki/YCbCr

.. autofunction:: rgb_to_ycbcr
.. autofunction:: ycbcr_to_rgb

.. autoclass:: YcbcrToRgb
.. autoclass:: RgbToYcbcr

YUV
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/YUV

.. autofunction:: rgb_to_yuv
.. autofunction:: yuv_to_rgb

.. autoclass:: RgbToYuv
.. autoclass:: YuvToRgb

XYZ
---

.. tip::
    Learn more: https://en.wikipedia.org/wiki/CIELUV

.. autofunction:: rgb_to_xyz
.. autofunction:: xyz_to_rgb

.. autoclass:: RgbToXyz
.. autoclass:: XyzToRgb
