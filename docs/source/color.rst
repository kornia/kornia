kornia.color
------------

.. currentmodule:: kornia.color

The functions in this section perform various color space conversions and intensity transformations.

Color Space Conversions
-----------------------

.. autofunction:: rgb_to_bgr
.. autofunction:: rgb_to_grayscale
.. autofunction:: rgb_to_hsv
.. autofunction:: rgb_to_hls
.. autofunction:: rgb_to_luv
.. autofunction:: rgb_to_rgba
.. autofunction:: rgb_to_xyz
.. autofunction:: rgb_to_ycbcr
.. autofunction:: rgb_to_yuv

.. autofunction:: rgba_to_rgb
.. autofunction:: rgba_to_bgr

.. autofunction:: bgr_to_grayscale
.. autofunction:: bgr_to_rgb
.. autofunction:: bgr_to_rgba

.. autofunction:: hls_to_rgb

.. autofunction:: hsv_to_rgb

.. autofunction:: luv_to_rgb

.. autofunction:: ycbcr_to_rgb
.. autofunction:: yuv_to_rgb

.. autofunction:: xyz_to_rgb


Intensity Transformations
-------------------------

.. autofunction:: adjust_brightness
.. autofunction:: adjust_contrast
.. autofunction:: adjust_gamma
.. autofunction:: adjust_hue
.. autofunction:: adjust_saturation
.. autofunction:: add_weighted

.. autofunction:: normalize
.. autofunction:: denormalize

.. autofunction:: histogram
.. autofunction:: histogram2d

Modules
-------

.. autoclass:: RgbToGrayscale
.. autoclass:: BgrToGrayscale
.. autoclass:: RgbToHsv
.. autoclass:: HsvToRgb
.. autoclass:: RgbToHls
.. autoclass:: HlsToRgb
.. autoclass:: RgbToBgr
.. autoclass:: BgrToRgb
.. autoclass:: RgbToYuv
.. autoclass:: YuvToRgb
.. autoclass:: RgbToRgba
.. autoclass:: BgrToRgba
.. autoclass:: RgbaToRgb
.. autoclass:: RgbaToBgr
.. autoclass:: RgbToXyz
.. autoclass:: XyzToRgb
.. autoclass:: RgbToLuv
.. autoclass:: LuvToRgb
.. autoclass:: YcbcrToRgb
.. autoclass:: RgbToYcbcr

.. autoclass:: Normalize
.. autoclass:: Denormalize

.. autoclass:: AdjustBrightness
.. autoclass:: AdjustContrast
.. autoclass:: AdjustSaturation
.. autoclass:: AdjustHue
.. autoclass:: AdjustGamma
.. autoclass:: AddWeighted
