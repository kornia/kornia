Equalization and histograms
===========================

.. currentmodule:: kornia.enhance

:func:`image_histogram2d` preserves leading batch and channel axes and reduces the final two
spatial axes. The KDE helpers :func:`histogram` and :func:`histogram2d` share one bin-center
grid across the batch.

Equalization
------------

:func:`equalize`, :func:`equalize_clahe` and :func:`equalize3d` expect a floating-point image in ``[0, 1]``
and raise :exc:`~kornia.core.exceptions.TypeCheckError` (a :exc:`TypeError`) for an integer or boolean image,
as the ``[0, 1]`` adjustment functions do; convert 8-bit data with ``image.float() / 255`` first.

.. autofunction:: equalize
.. autofunction:: equalize_clahe
.. autofunction:: equalize3d

Histograms
----------

.. autofunction:: histogram
.. autofunction:: histogram2d
.. autofunction:: image_histogram2d
