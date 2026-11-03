kornia.geometry
===============

.. meta::
   :description: The Kornia.geometry module provides essential geometric transformations for computer vision tasks, including 2D and 3D image manipulation. It includes submodules for image transforms, camera models, coordinate conversions, linear algebra operations, and depth map processing, supporting a wide range of geometric operations for accurate spatial transformations and 3D reconstructions.

Geometric image transformations are another key ingredient in computer vision for manipulating images.
Since geometry operations are typically performed in 2D or 3D, we provide algorithms for both cases.
This module, the original core of the library, groups its submodules by task:

- Warp, resize, crop, flip or register images, and build the matrices of those warps:
  :doc:`transform <geometry.transform>`.
- Convert between homogeneous and Euclidean points, pixel and normalized coordinates, rotation forms and pose
  conventions: :doc:`conversions <geometry.conversions>`.
- Apply a homogeneous transformation to points, compose, invert or relate ``4 x 4`` transformations, and measure
  distances of points to points and lines: :doc:`linalg <geometry.linalg>`. Grids of pixel coordinates:
  :doc:`grid <geometry.grid>`.
- Camera models, the projection and unprojection of points, and the Kannala-Brandt fisheye and affine distortion
  models on normalized points: :doc:`camera <geometry.camera>`; OpenCV (Brown-Conrady) lens distortion and
  undistortion of pixels and images, and pose from 2D-3D correspondences: :doc:`calibration <geometry.calibration>`.
- 3D points and surface normals from depth maps, and images warped by depth: :doc:`depth <geometry.depth>`.
- Two-view geometry, fundamental and essential matrices, relative pose and triangulation:
  :doc:`epipolar <geometry.epipolar>`; homographies from points or line segments:
  :doc:`homography <geometry.homography>`; any of these models robust to outliers: :doc:`ransac <geometry.ransac>`;
  the affine, null-vector and polynomial solvers they build on: :doc:`solvers <geometry.solvers>`.
- Rotations and rigid motions as objects that compose and invert: :doc:`liegroup <geometry.liegroup>`,
  :doc:`quaternion <geometry.quaternion>`, and poses between named frames, :doc:`pose <geometry.pose>`.
- Geometric primitives: 2D and 3D boxes, :doc:`boxes <geometry.boxes>`, with the tensor functions of
  :doc:`bbox <geometry.bbox>`; :doc:`keypoints <geometry.keypoints>`; lines and planes,
  :doc:`line <geometry.line>`; 2D and 3D vectors, :doc:`vector <geometry.vector>`.
- Sub-pixel peaks, soft argmax and non-maxima suppression of response maps: :doc:`subpix <geometry.subpix>`.
- Point clouds to and from PLY files: :doc:`pointcloud <geometry.pointcloud>`.

``kornia.geometry`` re-exports the names of every submodule here except ``boxes``, ``keypoints`` and ``vector``
(import those from their modules, such as ``kornia.geometry.boxes.Boxes``), and of ``quaternion`` only ``Quaternion``.

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
   pose <geometry.pose>
   quaternion <geometry.quaternion>
   line <geometry.line>
   bbox <geometry.bbox>
   boxes <geometry.boxes>
   keypoints <geometry.keypoints>
   vector <geometry.vector>
   subpix <geometry.subpix>
   solvers <geometry.solvers>
   grid <geometry.grid>
   pointcloud <geometry.pointcloud>
