Bayer RAW
=========

.. currentmodule:: kornia.color

.. tip::
    Learn more: https://en.wikipedia.org/wiki/Bayer_filter

Convert between a one-channel Bayer mosaic ``(*, 1, H, W)`` and an RGB image. The :class:`CFA` enum gives the
pattern of the mosaic.

Functions
---------

.. autofunction:: rgb_to_raw
.. autofunction:: raw_to_rgb
.. autofunction:: raw_to_rgb_2x2_downscaled

Modules
-------

.. autoclass:: CFA
    :members:
    :undoc-members:

.. autoclass:: RgbToRaw
.. autoclass:: RawToRgb
.. autoclass:: RawToRgb2x2Downscaled
