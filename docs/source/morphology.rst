kornia.morphology
======================

.. meta::
   :description: The Kornia.morphology module offers a set of morphological image processing operations such as dilation, erosion, opening, closing, gradient, top hat, bottom hat, and reconstruction. It enables users to apply these transformations to images for advanced computer vision tasks.

.. currentmodule::  kornia.morphology

Dilate, erode, open and close images ``(B, C, H, W)`` by a structuring element, and reconstruct them from a seed.
Every function applies one 2D ``kernel`` to every channel: its non-zero cells form the neighborhood, and a
``structuring_element`` of the same shape makes the operation non-flat. :func:`dilation` reflects the kernel and
:func:`erosion` does not, as in ``scipy.ndimage``; :doc:`Conventions & Pitfalls </get-started/conventions>` maps
the kernels and the border modes onto scipy, scikit-image and OpenCV.

- Grow bright regions with :func:`dilation` and shrink them with :func:`erosion`.
- Remove the bright details the kernel does not fit in with :func:`opening`, and fill the dark ones with
  :func:`closing`; :func:`top_hat` and :func:`bottom_hat` return those details alone.
- Measure edge strength, the dilation minus the erosion, with :func:`gradient`.
- Grow a seed image inside a mask until it stops changing with :func:`reconstruction`.

.. autofunction:: dilation
.. autofunction:: erosion
.. autofunction:: opening
.. autofunction:: closing
.. autofunction:: gradient
.. autofunction:: top_hat
.. autofunction:: bottom_hat
.. autofunction:: reconstruction
