Thresholding
============

.. currentmodule:: kornia.filters

:func:`in_range` keeps the pixels whose channels lie between bounds you give. :func:`otsu_threshold` picks one
threshold per image plane from its histogram and keeps the pixels above it.

.. autofunction:: in_range

.. autoclass:: InRange

.. autofunction:: otsu_threshold

.. autoclass:: OtsuThreshold
