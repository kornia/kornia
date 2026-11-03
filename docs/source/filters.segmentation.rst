Thresholding
============

.. currentmodule:: kornia.filters

Simple intensity-based segmentation: keep the pixels inside a value range, or split an image at Otsu's threshold.

.. autofunction:: in_range

.. autoclass:: InRange
.. autofunction:: otsu_threshold

.. autoclass:: OtsuThreshold

Local document binarization
---------------------------

Sauvola uses a different threshold at each pixel, based on the mean and standard
deviation in its neighborhood. This is useful for dark text on pages with shadows
or uneven lighting, where a global threshold can lose faint characters.

.. code-block:: python

    import torch
    from kornia.filters import sauvola_threshold

    # Float grayscale images should normally be scaled to [0, 1].
    image = torch.rand(2, 1, 128, 128)
    bright, threshold = sauvola_threshold(image, window_size=31)
    text = ~bright  # True for dark pixels, including equality with the threshold.

The mask is a hard comparison; use the returned threshold map when gradients
through the local statistics are needed. The operation treats channels independently;
convert RGB pages to grayscale before detecting text. For uint8 pages the default
contrast scale is 127.5; for normalized floating pages it is 0.5.

.. autofunction:: sauvola_threshold

.. autoclass:: SauvolaThreshold
