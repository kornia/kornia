kornia.image
============

.. meta::
   :description: The kornia.image module offers a high-level API designed for processing images in computer vision tasks. It provides functionalities for handling image size, pixel formats, channel orders, and image layouts, streamlining the manipulation of images in deep learning workflows. With a user-friendly interface, this module simplifies image data preprocessing and handling for various computer vision and machine learning tasks.

.. currentmodule:: kornia.image

A high-level API to describe and handle images: a typed image container, drawing primitives, conversions
between tensors and NumPy arrays, and terminal rendering.

.. list-table::
   :widths: 30 70

   * - :doc:`image.container`
     - :class:`Image`, :class:`ImageSize`, :class:`PixelFormat`, :class:`ChannelsOrder` and :class:`ImageLayout`.
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
