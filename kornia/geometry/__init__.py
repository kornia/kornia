# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r"""Geometric computer vision: warp images, model cameras, estimate two-view geometry, represent 3D.

The subpackages share kornia's conventions: images are ``(B, C, H, W)``, points and vectors sit in the last
dimension, pixel coordinates are ``(x, y)`` with ``x`` the column and pixel centres at integers, matrices act on
column vectors from the left, and camera extrinsics are world-to-camera, :math:`X_{cam} = R X + t`.
:doc:`/get-started/conventions` and :doc:`/get-started/camera-conventions` give the details. ``kornia.geometry``
re-exports the names of every subpackage below except ``boxes``, ``keypoints`` and ``vector`` (import those from
their modules, such as ``kornia.geometry.boxes.Boxes``), and of ``quaternion`` only ``Quaternion``.

Which subpackage to use:

- Warp, resize, crop, flip or register images, and build the matrices of those warps: ``transform``
  (:func:`~kornia.geometry.transform.warp_perspective`, :func:`~kornia.geometry.transform.warp_affine`,
  :func:`~kornia.geometry.transform.resize`).
- Convert between homogeneous and Euclidean points, pixel and normalized coordinates, rotation forms and pose
  conventions: ``conversions``.
- Apply a homogeneous ``(D + 1, D + 1)`` transformation to points; compose, invert or relate ``4 x 4``
  transformations; distances of points to points and lines: ``linalg``. Grids of pixel coordinates: ``grid``.
- Camera models, and the projection and unprojection of points: ``camera``
  (:class:`~kornia.geometry.camera.pinhole.PinholeCamera`); lens distortion and pose from 2D-3D correspondences:
  ``calibration``.
- 3D points and surface normals from depth maps, and images warped by depth: ``depth``.
- Two-view geometry, fundamental and essential matrices, relative pose and triangulation: ``epipolar``;
  homographies from points or line segments: ``homography``; any of these models robust to outliers: ``ransac``
  (:class:`~kornia.geometry.ransac.RANSAC`); the affine, null-vector and polynomial solvers they build on:
  ``solvers``.
- Rotations and rigid motions as objects that compose and invert: ``liegroup`` (:class:`~kornia.geometry.liegroup.So2`,
  :class:`~kornia.geometry.liegroup.So3`, :class:`~kornia.geometry.liegroup.Se2`,
  :class:`~kornia.geometry.liegroup.Se3`), ``quaternion`` (:class:`~kornia.geometry.quaternion.Quaternion`), and
  poses between named frames, ``pose`` (:class:`~kornia.geometry.pose.NamedPose`).
- Geometric primitives: 2D and 3D boxes, ``boxes`` (:class:`~kornia.geometry.boxes.Boxes`), with the tensor
  functions of ``bbox``; keypoints, ``keypoints``; lines, ``line``; planes, ``plane``; 2D and 3D vectors, ``vector``.
- Sub-pixel peaks, soft argmax and non-maxima suppression of response maps: ``subpix``.
- Point clouds to and from PLY files: ``pointcloud``.
"""

from .bbox import *
from .calibration import *
from .camera import *
from .conversions import *
from .depth import *
from .epipolar import *
from .grid import *
from .homography import *
from .liegroup import *
from .linalg import *
from .line import *
from .plane import *
from .pointcloud import *
from .pose import *
from .quaternion import Quaternion
from .ransac import *
from .solvers import *
from .subpix import *
from .transform import *
