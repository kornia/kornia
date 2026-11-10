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

"""Model cameras as a projection followed by a lens distortion, on ``Vector3`` points and flat parameter vectors.

This API is experimental. Points are :class:`~kornia.geometry.vector.Vector3` in the camera frame and pixels are
:class:`~kornia.geometry.vector.Vector2` with integer pixel centres, as on
:doc:`camera and world conventions </get-started/camera-conventions>`; depth is the camera-frame ``z``. ``params`` is
a flat ``(N,)`` or batched ``(B, N)`` vector that starts with ``[fx, fy, cx, cy]``, and its length is fixed by the
model.

Which class to use:

- A built-in model chosen by name: :class:`CameraModel`, with a :class:`CameraModelType`.
- The built-in models as classes: :class:`PinholeModel` (no lens distortion), :class:`BrownConradyModel` (OpenCV's
  radial and tangential distortion), :class:`KannalaBrandtK3` (fisheye) and :class:`Orthographic` (parallel
  projection).
- A custom model from a distortion and a projection: :class:`CameraModelBase`, with the classes of
  ``kornia.sensors.camera.distortion_model`` and ``kornia.sensors.camera.projection_model``.
- Cameras with extrinsics, on plain tensors and applied per camera in a batch:
  :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
"""

from .camera_model import (
    BrownConradyModel,
    CameraModel,
    CameraModelBase,
    CameraModelType,
    KannalaBrandtK3,
    Orthographic,
    PinholeModel,
)

__all__ = [
    "BrownConradyModel",
    "CameraModel",
    "CameraModelBase",
    "CameraModelType",
    "KannalaBrandtK3",
    "Orthographic",
    "PinholeModel",
]
