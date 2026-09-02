kornia.geometry
===============

.. meta::
   :description: The Kornia.geometry module provides essential geometric transformations for computer vision tasks, including 2D and 3D image manipulation. It includes submodules for image transforms, camera models, coordinate conversions, linear algebra operations, and depth map processing, supporting a wide range of geometric operations for accurate spatial transformations and 3D reconstructions.

Geometric image transformations are another key ingredient in computer vision for manipulating images.
Since geometry operations are typically performed in 2D or 3D, we provide algorithms for both cases.
This module, the original core of the library, consists of the following main submodules:

- :doc:`transform <geometry.transform>`: low-level interfaces to manipulate 2D images, with routines for rotating,
  scaling, translating and shearing; cropping functions in several modalities such as central crops and
  crop-and-resize; vertical and horizontal flips; resizing; functions to warp tensors given affine or
  perspective transformations; and utilities to compute the transformation matrices behind those operations.
- :doc:`camera <geometry.camera>`: routines specific to different camera representations, such as the pinhole
  or orthographic models, including projecting and unprojecting points between the camera and the world frame.
- :doc:`conversions <geometry.conversions>`: conversions between angle representations (radians and degrees),
  coordinate normalization, and homogeneous to Euclidean coordinates. It also includes conversions between 3D
  rotation representations: quaternions, axis-angle, rotation matrices and Euler angles.
- :doc:`linalg <geometry.linalg>`: general rigid-body homogeneous transformations, with functions to
  transform points between frames and to compose, invert and compute relative transformations.
- :doc:`depth <geometry.depth>`: layers to manipulate depth maps, such as computing 3D point clouds from depth maps
  and calibrated cameras, per-pixel surface normals, and warping frames given a calibrated camera setup.

The remaining submodules, listed below, cover bounding boxes and keypoints, calibration, epipolar geometry,
homographies, Lie groups, lines, point clouds, quaternions, RANSAC, polynomial solvers and sub-pixel refinement.

.. admonition:: About ``align_corners``

   ``align_corners`` is a switch offered by most PyTorch and Kornia geometric transform functions.
   Here is a simple illustration showing how a 4x4 image is upsampled to 8x8, made by
   `bkkm16 <https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9?u=ducha-aiki>`_.

   .. image:: https://user-images.githubusercontent.com/4803565/110627988-df8a4d00-81a2-11eb-8e13-06d3f7b09ef1.png
      :alt: Illustration of align_corners=True versus align_corners=False when upsampling a 4x4 image to 8x8

   - ``align_corners=True``: pixels are arranged as a grid of points. Points at the corners are aligned.
   - ``align_corners=False``: pixels are arranged as 1x1 areas. Area boundaries, rather than their centers, are aligned.

   The default is **not** uniform across the library; see the table in :doc:`get-started/conventions`.


.. currentmodule:: kornia.geometry

.. toctree::
   :maxdepth: 2

   transform <geometry.transform>
   camera <geometry.camera>
   conversions <geometry.conversions>
   linalg <geometry.linalg>
   depth <geometry.depth>
   epipolar <geometry.epipolar>
   homography <geometry.homography>
   ransac <geometry.ransac>
   calibration <geometry.calibration>
   liegroup <geometry.liegroup>
   quaternion <geometry.quaternion>
   line <geometry.line>
   bbox <geometry.bbox>
   boxes <geometry.boxes>
   keypoints <geometry.keypoints>
   subpix <geometry.subpix>
   solvers <geometry.solvers>
   grid <geometry.grid>
   pointcloud <geometry.pointcloud>
