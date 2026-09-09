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

from typing import Iterable, List, Union

import torch

from kornia.core.check import KORNIA_CHECK_SAME_DEVICE
from kornia.core.utils import _torch_inverse_cast
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.linalg import inverse_transformation, transform_points


class PinholeCamera:
    r"""Class that represents a Pinhole Camera model.

    Convention:
        - ``intrinsics`` is the :math:`(B, 4, 4)` calibration matrix whose top-left :math:`3 \times 3` block is
          ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, and ``extrinsics`` the :math:`(B, 4, 4)` **world-to-camera**
          transform ``[R | t]`` (OpenCV / COLMAP semantics): :meth:`project` takes **world** points, computes
          ``K (R X + t)`` and returns pixels, while :meth:`unproject` inverts that step -- it takes pixels and a
          camera-frame depth and returns **world** points. The functional API takes a ``K`` and no
          extrinsics, so it works in the **camera** frame:
          :func:`~kornia.geometry.camera.perspective.project_points` and
          :func:`~kornia.geometry.camera.perspective.unproject_points` take a :math:`(*, 3, 3)` ``K``, while
          :func:`~kornia.geometry.depth.depth_to_3d`, :func:`~kornia.geometry.depth.depth_to_3d_v2`,
          :func:`~kornia.geometry.depth.unproject_meshgrid` and
          :func:`~kornia.geometry.depth.depth_to_normals` need it **batched**, :math:`(B, 3, 3)`.
        - pixel coordinates are ``(u, v)`` = ``(x, y)`` = (column, row) with **integer pixel centres**: pixel
          ``(0, 0)`` is centred at ``(0, 0)``, which is what :func:`~kornia.geometry.grid.create_meshgrid`
          enumerates, so a centred image has its principal point at ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``.
          A half-pixel convention, which places the pixel *corner* at the origin (COLMAP), reports the same
          principal point half a pixel larger on each axis. See :doc:`/get-started/camera-conventions`.
        - ``depth`` is the camera-frame ``z`` coordinate. The ``normalize`` argument of
          :func:`~kornia.geometry.camera.perspective.unproject_points` and the ``normalize_points`` flags of
          :func:`~kornia.geometry.depth.depth_to_3d` and :func:`~kornia.geometry.depth.depth_to_3d_v2` read it
          as the Euclidean ray length instead, so the unprojected point has that norm rather than that ``z``.
        - the class stores the tensors it is constructed from instead of copying them, so :meth:`scale_` and
          the ``tx`` / ``ty`` / ``tz`` setters write into the caller's tensors. :meth:`scale` is the exception:
          it returns a new camera that owns both its ``intrinsics`` and its ``extrinsics``. :meth:`clone` is
          the deep copy of an existing camera.

    .. warning::
        :meth:`scale` and :meth:`scale_` rescale the principal point as ``cx' = s * cx`` â€” the half-pixel rule â€”
        which disagrees with the integer pixel centres above; it is tracked as a coordinated repair in
        `#4263 <https://github.com/kornia/kornia/issues/4263>`_. The write-through to the caller's tensors
        that remains on :meth:`scale_` and the setters is
        `#4264 <https://github.com/kornia/kornia/issues/4264>`_, the in-place :meth:`scale_` failure on an
        integer ``height`` / ``width`` with a floating-point scale factor
        `#4265 <https://github.com/kornia/kornia/issues/4265>`_, the batch-size and point-shape limitations
        `#4266 <https://github.com/kornia/kornia/issues/4266>`_. The behaviour described here is
        documented as it is; the issues above track the repairs.
