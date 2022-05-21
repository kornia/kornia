kornia.geometry.subpix
======================

.. currentmodule:: kornia.geometry.subpix

Module with useful functionalities to extract coordinates sub-pixel accuracy.

Convolutional
-------------

.. autofunction:: conv_soft_argmax2d
.. autofunction:: conv_soft_argmax3d
.. autofunction:: conv_quad_interp3d

Spatial
-------

.. autofunction:: spatial_softmax2d
.. autofunction:: spatial_expectation2d
.. autofunction:: spatial_soft_argmax2d
.. autofunction:: render_gaussian2d

Non Maxima Suppression
----------------------

.. autofunction:: nms2d
.. autofunction:: nms3d

Module
------

.. autoclass:: SpatialSoftArgmax2d
.. autoclass:: ConvSoftArgmax2d
.. autoclass:: ConvSoftArgmax3d
.. autoclass:: ConvQuadInterp3d
.. autoclass:: NonMaximaSuppression2d
.. autoclass:: NonMaximaSuppression3d
