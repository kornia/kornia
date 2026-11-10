kornia.image
============

.. meta::
   :description: The kornia.image module holds, draws on, converts and prints images: a checked image container, drawing primitives, conversions between channels-first tensors and channels-last NumPy arrays, image grids and terminal rendering.

.. currentmodule:: kornia.image

Hold, draw on, convert and print images: a checked image container, drawing primitives, conversions between
channels-first tensors and channels-last NumPy arrays, grids and terminal rendering. Image files are read and written
by :doc:`kornia.io <io>`.

.. list-table::
   :widths: 30 70

   * - :doc:`image.container`
     - :class:`Image`, :class:`ImageSize`, :class:`PixelFormat`, :class:`ColorSpace`, :class:`ChannelsOrder` and :class:`ImageLayout`.
   * - :doc:`image.drawing`
     - Draw lines, rectangles, convex polygons and points on image tensors.
   * - :doc:`image.conversion`
     - Convert between NumPy ``(H, W, C)`` arrays and ``(C, H, W)`` tensors.
   * - :doc:`image.printing`
     - Render an image tensor as text in a terminal.
   * - :doc:`image.utilities`
     - Grids and shape-preserving decorators.

.. toctree::
   :hidden:

   image container <image.container>
   drawing <image.drawing>
   conversion <image.conversion>
   printing <image.printing>
   utilities <image.utilities>
