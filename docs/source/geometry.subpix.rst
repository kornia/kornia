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
.. autofunction:: iterative_quad_interp3d

.. tip::

   :class:`AdaptiveQuadInterp3d` (the default subpix module in
   :class:`~kornia.feature.ScaleSpaceDetector`) automatically picks the faster
   backend based on the input device:

   * **CUDA** → :func:`conv_quad_interp3d` — batched gather+solve, 1.5–2× faster on GPU.
   * **CPU**  → :func:`iterative_quad_interp3d` — processes only NMS maxima directly,
     no dilation overhead.

   Both backends produce numerically identical results (max difference < 2 × 10\ :sup:`-6`).

   .. code-block:: python

      import torch
      from kornia.geometry.subpix import AdaptiveQuadInterp3d
      from kornia.feature import ScaleSpaceDetector
      from kornia.feature.responses import BlobDoG
      from kornia.geometry.transform import ScalePyramid

      detector = ScaleSpaceDetector(
          num_features=2000,
          resp_module=BlobDoG(),
          # default — auto-selects conv on CUDA, patch on CPU:
          subpix_module=AdaptiveQuadInterp3d(strict_maxima_bonus=0.0),
          scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
          scale_space_response=True,
          minima_are_also_good=True,
      )

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
.. autofunction:: nms3d_minmax

Module
------

.. autoclass:: SpatialSoftArgmax2d
.. autoclass:: ConvSoftArgmax2d
.. autoclass:: ConvSoftArgmax3d
.. autoclass:: AdaptiveQuadInterp3d
.. autoclass:: ConvQuadInterp3d
.. autoclass:: IterativeQuadInterp3d
.. autoclass:: NonMaximaSuppression2d
.. autoclass:: NonMaximaSuppression3d
