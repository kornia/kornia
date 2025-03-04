kornia.geometry.subpix
======================

.. meta::
   :name: description
   :content: "The kornia.geometry.subpix module provides functionalities for extracting coordinates with sub-pixel accuracy. It includes convolutional methods like soft argmax and quadratic interpolation for precise 2D and 3D coordinate extraction. Additionally, it offers spatial softmax and expectation techniques, as well as non-maximum suppression (NMS) for 2D and 3D data, making it ideal for tasks requiring high-resolution spatial localization in computer vision."

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
