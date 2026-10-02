Adjustment
==========

.. currentmodule:: kornia.enhance

Hue and saturation adjustments and :func:`shift_rgb` take RGB channels at axis -3. Functions
whose name ends in ``_raw`` take HSV instead; their RGB counterparts perform that conversion.
Brightness, gamma, logarithmic and sigmoid adjustments, :func:`adjust_contrast`, and
:func:`threshold` are elementwise and accept any channel count. :func:`sharpness` filters
neighboring pixels, :func:`equalize` uses each
channel's histogram, and :func:`adjust_contrast_with_mean_subtraction` uses an image mean.

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
