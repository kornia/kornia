Equalization and histograms
===========================

.. currentmodule:: kornia.enhance

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

The three histograms are kernel estimates and differ in their input:

- :func:`histogram`: rows of samples ``(B, D)``, giving one normalized histogram per row.
- :func:`histogram2d`: two rows of samples, giving their normalized joint histogram.
- :func:`image_histogram2d`: an image, giving one histogram of the pixel values per channel.

.. autofunction:: histogram
.. autofunction:: histogram2d
.. autofunction:: image_histogram2d
