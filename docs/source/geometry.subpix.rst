kornia.geometry.subpix
======================

.. meta::
   :description: The kornia.geometry.subpix module provides functionalities for extracting coordinates with sub-pixel accuracy. It includes convolutional methods like soft argmax and quadratic interpolation for precise 2D and 3D coordinate extraction. Additionally, it offers spatial softmax and expectation techniques, as well as non-maximum suppression (NMS) for 2D and 3D data, making it ideal for tasks requiring high-resolution spatial localization in computer vision.

.. currentmodule:: kornia.geometry.subpix

Sub-pixel locations from heatmaps and scale-space volumes. :func:`spatial_soft_argmax2d` returns one ``(x, y)`` per
heatmap channel, the softmax-weighted mean over the whole map, and :func:`conv_soft_argmax2d` and
:func:`conv_soft_argmax3d` one per sliding window. :func:`nms2d`, :func:`nms3d` and :func:`nms3d_minmax` keep the strict
local maxima, and :func:`conv_quad_interp3d` and :func:`iterative_quad_interp3d` refine the maxima of a scale-space
volume by a quadratic fit, with :class:`AdaptiveQuadInterp3d` picking the faster one for the device. Coordinates are
``(x, y)`` in 2D and ``(d, x, y)`` in 3D; see :ref:`Coordinates and sizes <coordinate-conventions>`. Each class under
Modules except :class:`AdaptiveQuadInterp3d` stores the arguments of its function form.

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

   With ``max_candidates`` unset, the backends agree on every maximum whose centre stays within
   ``dilation_radius`` voxels (default 1) of where it started. Beyond that :func:`conv_quad_interp3d` can reject a
   maximum that :func:`iterative_quad_interp3d` still refines, so CPU and CUDA keypoints can differ;
   ``dilation_radius=n_iters`` makes the backends agree.

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
          subpix_module=AdaptiveQuadInterp3d(),
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

Non-Maxima Suppression
----------------------

.. autofunction:: nms2d
.. autofunction:: nms3d
.. autofunction:: nms3d_minmax

Modules
-------

.. autoclass:: SpatialSoftArgmax2d
.. autoclass:: ConvSoftArgmax2d
.. autoclass:: ConvSoftArgmax3d
.. autoclass:: AdaptiveQuadInterp3d
.. autoclass:: ConvQuadInterp3d
.. autoclass:: IterativeQuadInterp3d
.. autoclass:: NonMaximaSuppression2d
.. autoclass:: NonMaximaSuppression3d
