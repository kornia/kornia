kornia.image
============

.. meta::
   :name: description
   :content: "The kornia.image module offers a high-level API designed for processing images in computer vision tasks. It provides functionalities for handling image size, pixel formats, channel orders, and image layouts, streamlining the manipulation of images in deep learning workflows. With a user-friendly interface, this module simplifies image data preprocessing and handling for various computer vision and machine learning tasks."

Module to provide a high level API to process images.

.. currentmodule:: kornia.image

.. autoclass:: ImageSize
    :members:
    :undoc-members:

.. autoclass:: PixelFormat
    :members:
    :undoc-members:

.. autoclass:: ChannelsOrder
    :members:
    :undoc-members:

.. autoclass:: ImageLayout
   :members:
   :undoc-members:

.. autoclass:: Image
    :members:
    :undoc-members:

Drawing
-------

.. autofunction:: draw_line
.. autofunction:: draw_rectangle
.. autofunction:: draw_convex_polygon
.. autofunction:: draw_point2d

Image Conversion
----------------

.. autofunction:: tensor_to_image
.. autofunction:: image_to_tensor
.. autofunction:: image_list_to_tensor
.. autoclass:: ImageToTensor

Image Printing
--------------

.. autofunction:: image_to_string
.. autofunction:: print_image

Utilities
---------

.. autofunction:: make_grid
.. autofunction:: perform_keep_shape_image
.. autofunction:: perform_keep_shape_video
