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

r"""Lie groups for 2D and 3D rotations and rigid motions: composition, point maps, exp and log.

Each class holds one group element or a batch of ``B`` as an ``nn.Module``, so ``state_dict()`` and ``.to()`` reach
its tensors. ``a * b`` composes as ``a.matrix() @ b.matrix()``, so ``b`` acts first, and ``g * p`` maps points as
:math:`R p + t`. ``exp`` maps a tangent vector to the group, and ``log`` maps back to the principal vector, whose
rotation angle is at most :math:`\pi` in magnitude. Tangent vectors put the rotation last, in radians: the angle
``theta`` for ``So2``, the rotation vector ``omega`` for ``So3``, ``(v_x, v_y, theta)`` for ``Se2`` and
``[upsilon, omega]`` for ``Se3``; ``hat`` turns one into its matrix and ``vee`` reads it back. The conventions page
of the documentation compares these with scipy, Sophus and Eigen.

Which class to use:

- A 2D rotation: :class:`So2`, stored as a complex number. A 2D rotation and translation: :class:`Se2`.
- A 3D rotation: :class:`So3`, stored as a :class:`~kornia.geometry.quaternion.Quaternion`, with the Jacobians of
  its ``exp``. A 3D rotation and translation: :class:`Se3`.
- A rigid pose that names its source and destination frames and checks them when composing:
  :class:`~kornia.geometry.pose.NamedPose`. Plain ``(N, 4, 4)`` matrices:
  :func:`~kornia.geometry.linalg.compose_transformations`.
- Quaternion arithmetic, interpolation and averaging, without acting on points:
  :class:`~kornia.geometry.quaternion.Quaternion` and :func:`~kornia.geometry.quaternion.average_quaternions`.
"""

from .se2 import Se2
from .se3 import Se3
from .so2 import So2
from .so3 import So3

__all__ = ["Se2", "Se3", "So2", "So3"]
