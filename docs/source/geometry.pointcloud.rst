kornia.geometry.pointcloud
==========================

.. meta::
   :description: The kornia.geometry.pointcloud module provides functionality for loading and saving point clouds in PLY format. It includes functions for reading and writing 3D point cloud data, making it easy to work with point cloud files in computer vision and geometric processing tasks.

.. currentmodule:: kornia.geometry.pointcloud

Point clouds to and from PLY files. :func:`save_pointcloud_ply` and :func:`save_pointcloud_ply_binary` write the
``x``, ``y`` and ``z`` of a channels-last ``(*, 3)`` tensor as ``double`` vertices, as text or as little-endian
binary; :func:`load_pointcloud_ply` and :func:`load_pointcloud_ply_binary` read the vertex coordinates of an ASCII or a
binary file, skipping normals, colours and faces, into a float32 ``(N, 3)`` tensor.

.. autofunction:: save_pointcloud_ply
.. autofunction:: load_pointcloud_ply
.. autofunction:: save_pointcloud_ply_binary
.. autofunction:: load_pointcloud_ply_binary
