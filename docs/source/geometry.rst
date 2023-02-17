kornia.geometry
===============

Geometric image transformations is another key ingredient in computer vision to manipulate images.
Since geometry operations are typically performed in 2D or 3D, we provide several algorithms to work
with both cases. This module, the original core of the library, consists of the following submodules:
transforms, camera, conversions, linalg and depth. We next describe each of them:

- `transforms`: The module provides low level interfaces to manipulate 2D images, with routines for Rotating,
  Scaling, Translating, Shearing; Cropping functions in several modalities such as central crops,
  crop and resize; Flipping transformations in the vertical and horizontal axis; Resizing operations;
  Functions to warp tensors given affine or perspective transformations,
  and utilities to compute the transformation matrices to perform the mentioned operations.
- `camera`: A set of routines specific to different types of camera representations such as Pinhole
  or Orthographic models containing functionalities such as projecting and unprojecting points from the
  camera to a world frame.
- `conversions`: Routines to perform conversions between angle representation such as
  radians to degrees, coordinates normalization, and homogeneous to euclidean. Moreover, we include advanced
  conversions for 3D geometry representations such as Quaternion, Axis-Angle, Rotation Matrix, or Rodrigues
  formula.
- `linalg`: Functions to perform general rigid-body homogeneous transformations. We include implementations to
  transform points between frames and for homogeneous transformations, manipulation such as composition,
  inverse and to compute relative poses.
- `depth`: A set of layers to manipulate depth maps such as how to compute 3D point clouds given depth maps and
  calibrated cameras; compute surface normals per pixel and warp tensor frames given calibrated cameras setup.


:Resources:

   **align_corners**

   align_corners is a switch that widely offered in PyTorch geometric transform functions.
   Here is a simple illustration showing how a 4x4 image is upsampled to 8x8, made by
   `bkkm16 <https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9?u=ducha-aiki>`_.

   .. image:: https://user-images.githubusercontent.com/4803565/110627988-df8a4d00-81a2-11eb-8e13-06d3f7b09ef1.png

   - `align_corners=True`, pixels are arranged as a grid of points. Points at the corners are aligned.
   - `align_corners=False`, pixels are arranged as 1x1 areas. Area boundaries, rather than their centers, are aligned.


.. currentmodule:: kornia.geometry

.. toctree::
   :maxdepth: 3

   geometry.bbox
   geometry.boxes
   geometry.calibration
   geometry.camera
   geometry.conversions
   geometry.depth
   geometry.epipolar
   geometry.homography
   geometry.liegroup
   geometry.linalg
   geometry.line
   geometry.quaternion
   geometry.subpix
   geometry.transform
   geometry.ransac
