Equalization and histograms
===========================

.. currentmodule:: kornia.enhance

Image histogram functions preserve leading batch and channel axes and reduce the
final two spatial axes. The KDE helpers use a shared bin-center grid per batch.

Equalization
------------

.. autofunction:: equalize
.. autofunction:: equalize_clahe
.. autofunction:: equalize3d

Histograms
----------

.. autofunction:: histogram
.. autofunction:: histogram2d
.. autofunction:: image_histogram2d
