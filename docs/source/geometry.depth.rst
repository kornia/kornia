kornia.geometry.depth
=====================

.. meta::
   :name: description
   :content: "The kornia.geometry.depth module provides functions for working with depth-related transformations in 3D vision tasks. Key functionalities include computing depth from disparity, converting depth maps to 3D points, obtaining surface normals from depth data, and unprojecting depth data from mesh grids. Additionally, the module supports depth-based image warping and working with depth through plane equations, enabling advanced geometric operations in computer vision."

.. currentmodule:: kornia.geometry.depth

.. autofunction:: depth_from_disparity
.. autofunction:: depth_to_3d
.. autofunction:: depth_to_3d_v2
.. autofunction:: unproject_meshgrid
.. autofunction:: depth_to_normals
.. autofunction:: depth_from_plane_equation
.. autofunction:: warp_frame_depth
