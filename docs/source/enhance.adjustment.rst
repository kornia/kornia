Adjustment
==========

.. currentmodule:: kornia.enhance

Hue and saturation adjustments and :func:`shift_rgb` take RGB channels at axis -3. Functions
whose name ends in ``_raw`` take HSV instead; their RGB counterparts perform that conversion.
The other functions accept any channel count. :func:`sharpness` filters neighboring pixels and
:func:`adjust_contrast_with_mean_subtraction` subtracts an image mean, so their output also
depends on the rest of the image.

Three adjustments have two definitions, and the second of each pair matches PIL's ``ImageEnhance``:

.. list-table::
   :header-rows: 1

   * - Adjustment
     - Direct
     - Matches PIL
   * - Brightness
     - :func:`adjust_brightness` adds the factor.
     - :func:`adjust_brightness_accumulative` multiplies by it.
   * - Contrast
     - :func:`adjust_contrast` multiplies by the factor.
     - :func:`adjust_contrast_with_mean_subtraction` blends with the mean gray level.
   * - Saturation
     - :func:`adjust_saturation` scales the HSV saturation.
     - :func:`adjust_saturation_with_gray_subtraction` blends with the gray image.

The brightness, contrast, gamma, logarithmic and sigmoid adjustments,
:func:`adjust_brightness_accumulative`, :func:`adjust_contrast_with_mean_subtraction`,
:func:`adjust_saturation_with_gray_subtraction`, :func:`posterize`, :func:`solarize`,
:func:`sharpness` and :func:`shift_rgb` expect a floating-point image in ``[0, 1]`` and raise
:exc:`~kornia.core.exceptions.TypeCheckError` (a :exc:`TypeError`) for an integer image; convert
8-bit data with ``image.float() / 255`` first. :func:`adjust_hue`, :func:`adjust_saturation`,
:func:`add_weighted` and :func:`invert` (with ``max_val``) also accept integer input.

Functions
---------

.. autofunction:: add_weighted
.. autofunction:: adjust_brightness
.. autofunction:: adjust_brightness_accumulative
.. autofunction:: adjust_contrast
.. autofunction:: adjust_contrast_with_mean_subtraction
.. autofunction:: adjust_gamma
.. autofunction:: adjust_hue
.. autofunction:: adjust_hue_raw
.. autofunction:: adjust_saturation
.. autofunction:: adjust_saturation_raw
.. autofunction:: adjust_saturation_with_gray_subtraction
.. autofunction:: adjust_sigmoid
.. autofunction:: adjust_log
.. autofunction:: invert
.. autofunction:: posterize
.. autofunction:: sharpness
.. autofunction:: shift_rgb
.. autofunction:: solarize
.. autofunction:: threshold

Modules
-------

.. autoclass:: AdjustBrightness
.. autoclass:: AdjustBrightnessAccumulative
.. autoclass:: AdjustContrast
.. autoclass:: AdjustContrastWithMeanSubtraction
.. autoclass:: AdjustSaturation
.. autoclass:: AdjustSaturationWithGraySubtraction
.. autoclass:: AdjustHue
.. autoclass:: AdjustGamma
.. autoclass:: AdjustSigmoid
.. autoclass:: AdjustLog
.. autoclass:: AddWeighted
.. autoclass:: Invert
.. autoclass:: Threshold
.. autoclass:: ThresholdType
