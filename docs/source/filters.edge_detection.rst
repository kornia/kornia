Edge detection
==============

.. currentmodule:: kornia.filters

:func:`sobel` returns the gradient magnitude of an image, :func:`spatial_gradient` its signed x and y derivatives (or
second derivatives), :func:`laplacian` a second-derivative response and :func:`canny` a thin binary edge map.
:func:`spatial_gradient3d` differentiates volumes.

Functions
---------

.. autofunction:: canny
.. autofunction:: laplacian
.. autofunction:: sobel
.. autofunction:: spatial_gradient
.. autofunction:: spatial_gradient3d

Modules
-------

.. autoclass:: Laplacian
.. autoclass:: Sobel
.. autoclass:: Canny
.. autoclass:: SpatialGradient
.. autoclass:: SpatialGradient3d
