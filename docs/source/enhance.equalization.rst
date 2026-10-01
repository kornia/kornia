Equalization and histograms
===========================

.. currentmodule:: kornia.enhance

:func:`image_histogram2d` preserves leading batch and channel axes and reduces the final two
spatial axes. The KDE helpers :func:`histogram` and :func:`histogram2d` share one bin-center
grid across the batch.

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
