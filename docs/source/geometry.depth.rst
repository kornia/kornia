kornia.geometry.depth
=====================

.. meta::
   :description: The kornia.geometry.depth module provides functions for working with depth-related transformations in 3D vision tasks. Key functionalities include computing depth from disparity, converting depth maps to 3D points, obtaining surface normals from depth data, and unprojecting depth data from mesh grids. Additionally, the module supports depth-based image warping and working with depth through plane equations, enabling advanced geometric operations in computer vision.

Unproject depth maps to 3D points with :func:`~kornia.geometry.depth.depth_to_3d` (channels first) or
:func:`~kornia.geometry.depth.depth_to_3d_v2` (channels last, with rays cached by
:func:`~kornia.geometry.depth.unproject_meshgrid`), or to surface normals with
:func:`~kornia.geometry.depth.depth_to_normals`. Get depth from a stereo disparity map with
:func:`~kornia.geometry.depth.depth_from_disparity`, or for pixels on a known plane with
:func:`~kornia.geometry.depth.depth_from_plane_equation`. Render an image from another viewpoint by depth with
:func:`~kornia.geometry.depth.warp_frame_depth` (a 4x4 pose and one camera matrix), or with
:class:`~kornia.geometry.depth.DepthWarper` and :func:`~kornia.geometry.depth.depth_warp` (two pinhole camera models).

.. currentmodule:: kornia.geometry.depth

.. autofunction:: depth_from_disparity
.. autofunction:: depth_to_3d
.. autofunction:: depth_to_3d_v2
.. autofunction:: unproject_meshgrid
.. autofunction:: depth_to_normals
.. autofunction:: depth_from_plane_equation
.. autofunction:: warp_frame_depth
.. autofunction:: depth_warp

.. autoclass:: DepthWarper
    :members:
