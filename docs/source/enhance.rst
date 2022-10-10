kornia.enhance
==============

.. currentmodule:: kornia.enhance

The functions in this section perform normalisations and intensity transformations.

Adjustment
----------

.. autofunction:: add_weighted
.. autofunction:: adjust_brightness
.. autofunction:: adjust_contrast
.. autofunction:: adjust_contrast_with_mean_subtraction
.. autofunction:: adjust_gamma
.. autofunction:: adjust_hue
.. autofunction:: adjust_saturation
.. autofunction:: adjust_sigmoid
.. autofunction:: adjust_log
.. autofunction:: invert
.. autofunction:: posterize
.. autofunction:: sharpness
.. autofunction:: solarize

Interactive Demo
~~~~~~~~~~~~~~~~
.. raw:: html

    <gradio-app space="kornia/kornia-image-enhancement"></gradio-app>

Visit the demo on `Hugging Face Spaces <https://huggingface.co/spaces/kornia/kornia-image-enhancement>`_.

Equalization
------------

.. autofunction:: equalize
.. autofunction:: equalize_clahe
.. autofunction:: equalize3d

.. autofunction:: histogram
.. autofunction:: histogram2d
.. autofunction:: image_histogram2d

Normalizations
--------------

.. autofunction:: normalize
.. autofunction:: normalize_min_max
.. autofunction:: denormalize
.. autofunction:: zca_mean
.. autofunction:: zca_whiten
.. autofunction:: linear_transform


Modules
-------

.. autoclass:: Normalize
.. autoclass:: Denormalize
.. autoclass:: ZCAWhitening
    :members:

.. autoclass:: AdjustBrightness
.. autoclass:: AdjustContrast
.. autoclass:: AdjustSaturation
.. autoclass:: AdjustHue
.. autoclass:: AdjustGamma
.. autoclass:: AdjustSigmoid
.. autoclass:: AdjustLog
.. autoclass:: AddWeighted

.. autoclass:: Invert
