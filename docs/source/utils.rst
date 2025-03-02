kornia.utils
===================

.. meta::
   :name: description
   :content: "The `kornia.utils` module provides a variety of utility functions for computer vision tasks, including image manipulation, pointcloud handling, device management, memory operations, and automatic mixed precision support. These utilities streamline workflows for tasks like image transformation, grid generation, and point cloud file handling."

.. currentmodule:: kornia.utils

Draw
----

.. autofunction:: draw_line
.. autofunction:: draw_rectangle
.. autofunction:: draw_convex_polygon

Image
-----

.. autofunction:: tensor_to_image
.. autofunction:: image_to_tensor
.. autofunction:: image_list_to_tensor
.. autofunction:: image_to_string
.. autofunction:: print_image

Grid
----

.. autofunction:: create_meshgrid
.. autofunction:: create_meshgrid3d

Pointcloud
----------

.. autofunction:: save_pointcloud_ply
.. autofunction:: load_pointcloud_ply

Memory
-------

.. autofunction:: one_hot
.. autofunction:: batched_forward

Device
-------

.. autofunction:: get_cuda_device_if_available
.. autofunction:: get_mps_device_if_available
.. autofunction:: get_cuda_or_mps_device_if_available
.. autofunction:: map_location_to_cpu


Automatic Mixed Precision
-------------------------
.. autofunction:: is_autocast_enabled
