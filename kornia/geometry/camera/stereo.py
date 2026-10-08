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

from typing import Any

import torch

from kornia.core.utils import is_exporting
from kornia.geometry.grid import create_meshgrid
from kornia.geometry.linalg import transform_points


class StereoException(Exception):
    """Handle errors related to stereo camera calibration and rectification."""

    def __init__(self, msg: str, *args: Any, **kwargs: Any) -> None:
        r"""Construct custom exception for the :module:`~kornia.geometry.camera.stereo` module.

        Adds a general helper module redirecting the user to the proper documentation site.

        Args:
            msg: Custom message to add to the general message.
            *args: Additional argument passthrough
            **kwargs: Additional argument passthrough

        """
        doc_help = (
            "\n Please check documents here: "
            "https://kornia.readthedocs.io/en/latest/geometry.camera.stereo.html for further information and examples."
        )
        final_msg = msg + doc_help
        # type ignore because of mypy error:
        # Too many arguments for "__init__" of "BaseException"
        super().__init__(final_msg, *args, **kwargs)


class StereoCamera:
    """Represent a horizontal stereo camera setup for depth estimation.

    Convention:
        - the two arguments are the **rectified projection matrices** of the left and the right camera, each of
          shape :math:`(B, 3, 4)`: ``[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]`` for the left camera, and
          the same matrix with ``-tx * fx`` in the last column for the right one. The constructor requires the
          two to be equal outside that last column.
        - the baseline is read back as the attribute :attr:`tx` ``= -P_right[0, 3] / fx``, which must be
          strictly **positive** for every rig in the batch: a zero baseline and swapped cameras both raise. A
          non-finite ``P_right[0, 3]`` is not screened, and these checks and the equal-intrinsics check are
          skipped under ``torch.export``.
          The :doc:`/geometry.camera.stereo` page's symbol ``tx`` is the negation of this attribute; :attr:`Q`
          is the page's matrix evaluated at the page's ``tx``.
        - ``Q[0, 0]`` carries ``fy`` and ``Q[1, 1]`` carries ``fx``; with the divide by ``W = -fy * disparity``
          the result is ``X = (u - cx) Z / fx``, ``Y = (v - cy) Z / fy``, ``Z = fx * tx / disparity``.
        - ``u`` is the **column** index and ``v`` the **row** index, as in ``cv2.reprojectImageTo3D``.
        - a disparity map is channels-**last**, :math:`(B, H, W, 1)` (a :math:`(B, 1, H, W)` map is rejected),
          and the point cloud is :math:`(B, H, W, 3)`.
        - pixel centres are integers; see the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

    .. warning::
        A differing ``cx`` is **rejected**, although ``Q[3, 3] = fy * (cx_left - cx_right)`` exists for that
        case, so it is zero on every rig the constructor accepts outside ``torch.export``:
        `#4270 <https://github.com/kornia/kornia/issues/4270>`_. A zero disparity (a point at infinity) makes
        ``W = 0``; the divide is then skipped and a finite placeholder, behind the camera on a real rig, is
        returned and not flagged as invalid: `#4555 <https://github.com/kornia/kornia/issues/4555>`_.

    .. warning::
        The module-level :func:`~kornia.geometry.camera.stereo.reproject_disparity_to_3D` is rendered on
        :doc:`/geometry.camera.stereo` but is in no ``__all__``, so it is not reachable as
        ``kornia.geometry.reproject_disparity_to_3D``. Tracked as
        `#4275 <https://github.com/kornia/kornia/issues/4275>`_.

    Args:
        rectified_left_camera: The rectified left camera projection matrix of shape :math:`(B, 3, 4)`.
        rectified_right_camera: The rectified right camera projection matrix of shape :math:`(B, 3, 4)`.
    """

    def __init__(self, rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor) -> None:
        r"""Class representing a horizontal stereo camera setup.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape :math:`(B, 3, 4)`
            rectified_right_camera: The rectified right camera projection matrix
              of shape :math:`(B, 3, 4)`

        """
        self._check_stereo_camera(rectified_left_camera, rectified_right_camera)
        self.rectified_left_camera: torch.Tensor = rectified_left_camera
        self.rectified_right_camera: torch.Tensor = rectified_right_camera

        self.device = self.rectified_left_camera.device
        self.dtype = self.rectified_left_camera.dtype

        self._Q_matrix = self._init_Q_matrix()

    @staticmethod
    def _check_stereo_camera(rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor) -> None:
        r"""Ensure user specified correct camera matrices.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape :math:`(B, 3, 4)`
            rectified_right_camera: The rectified right camera projection matrix
              of shape :math:`(B, 3, 4)`

        """
        # Ensure correct shapes
        if len(rectified_left_camera.shape) != 3:
            raise StereoException(
                f"Expected 'rectified_left_camera' to have 3 dimensions. Got {rectified_left_camera.shape}."
            )

        if len(rectified_right_camera.shape) != 3:
            raise StereoException(
                f"Expected 'rectified_right_camera' to have 3 dimension. Got {rectified_right_camera.shape}."
            )

        if rectified_left_camera.shape[-2:] != (3, 4):
            raise StereoException(
                f"Expected each 'rectified_left_camera' to be of shape (3, 4). Got {rectified_left_camera.shape[-2:]}."
            )

        if rectified_right_camera.shape[-2:] != (3, 4):
            raise StereoException(
                "Expected each 'rectified_right_camera' to be of shape (3, 4). "
                f"Got {rectified_right_camera.shape[-2:]}."
            )

        # Ensure same devices for cameras.
        if rectified_left_camera.device != rectified_right_camera.device:
            raise StereoException(
                "Expected 'rectified_left_camera' and 'rectified_right_camera' "
                "to be on the same devices."
                f"Got {rectified_left_camera.device} and {rectified_right_camera.device}."
            )

        # Ensure same dtypes for cameras.
        if rectified_left_camera.dtype != rectified_right_camera.dtype:
            raise StereoException(
                "Expected 'rectified_left_camera' and 'rectified_right_camera' to"
                "have same dtype."
                f"Got {rectified_left_camera.dtype} and {rectified_right_camera.dtype}."
            )

        # Ensure all intrinsics parameters (fx, fy, cx, cy) are the same in both cameras.
        # The check reads the data, which graph capture cannot do; skip it under export.
        if not is_exporting() and not torch.all(
            torch.eq(rectified_left_camera[..., :, :3], rectified_right_camera[..., :, :3])
        ):
            raise StereoException(
                "Expected 'left_rectified_camera' and 'rectified_right_camera' to have"
                "same parameters except for the last column."
                f"Got {rectified_left_camera[..., :, :3]} and {rectified_right_camera[..., :, :3]}."
            )

        # Reject every rig whose baseline is zero or points the wrong way. The right camera's last column is
        # -tx * fx, so it must be strictly negative: zero means coincident cameras, which collapses Q and sends
        # every disparity to the origin, and positive means the two cameras are swapped. The quantifier is
        # ``any``, so one bad rig cannot hide behind good ones, and an empty batch stays vacuously valid. Neither
        # comparison screens non-finite input: ``nan`` fails both of them and ``-inf`` is negative, so both are
        # still accepted, exactly as they were before these checks were tightened.
        # The check reads the data, which graph capture cannot do; skip it under export.
        tx_fx = rectified_right_camera[..., 0, 3]
        if not is_exporting():
            if torch.any(tx_fx == 0):
                raise StereoException(
                    "Expected a non-zero stereo baseline, but :math:`T_x * f_x` is 0 for at least one camera pair, "
                    f"so `tx` is 0 and every disparity would reproject to the origin. Got {tx_fx}."
                )
            if torch.any(tx_fx > 0):
                raise StereoException(f"Expected :math:`T_x * f_x` to be negative for every camera pair. Got {tx_fx}.")

    @property
    def batch_size(self) -> int:
        r"""Return the batch size of the storage.

        Returns:
           scalar with the batch size

        """
        return self.rectified_left_camera.shape[0]

    @property
    def fx(self) -> torch.Tensor:
        r"""Return the focal length in the x-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        r"""Returns the focal length in the y-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx_left(self) -> torch.Tensor:
        r"""Return the x-coordinate of the principal point for the left camera.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cx_right(self) -> torch.Tensor:
        r"""Return the x-coordinate of the principal point for the right camera.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return self.rectified_right_camera[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        r"""Return the y-coordinate of the principal point.

        Note that the y-coordinate of the principal points
        is assumed to be equal for the left and right camera.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> torch.Tensor:
        r"""The horizontal baseline between the two cameras.

        Returns:
            torch.Tensor of shape :math:`(B)`

        """
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> torch.Tensor:
        r"""The Q matrix of the horizontal stereo setup.

        This matrix is used for reprojecting a disparity torch.Tensor to
        the corresponding point cloud. Note that this is in a general form that allows different focal
        lengths in the x and y direction.

        Return:
            The Q matrix of shape :math:`(B, 4, 4)`.

        """
        return self._Q_matrix

    def _init_Q_matrix(self) -> torch.Tensor:
        r"""Initialize the Q matrix of the horizontal stereo setup. See the Q property.

        Returns:
            The Q matrix of shape :math:`(B, 4, 4)`.

        """
        Q = torch.zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        baseline: torch.Tensor = -self.tx
        Q[:, 0, 0] = self.fy * baseline
        Q[:, 0, 3] = -self.fy * self.cx_left * baseline
        Q[:, 1, 1] = self.fx * baseline
        Q[:, 1, 3] = -self.fx * self.cy * baseline
        Q[:, 2, 3] = self.fx * self.fy * baseline
        Q[:, 3, 2] = -self.fy
        Q[:, 3, 3] = self.fy * (self.cx_left - self.cx_right)  # NOTE: This is usually zero.
        return Q

    def reproject_disparity_to_3D(self, disparity_tensor: torch.Tensor) -> torch.Tensor:
        r"""Reproject the disparity torch.Tensor to a 3D point cloud.

        See the Convention block on :class:`~kornia.geometry.camera.stereo.StereoCamera`.

        Args:
            disparity_tensor: Disparity torch.Tensor of shape :math:`(B, H, W, 1)`.

        Returns:
            The 3D point cloud of shape :math:`(B, H, W, 3)`

        """
        return reproject_disparity_to_3D(disparity_tensor, self.Q)


def _check_disparity_tensor(disparity_tensor: torch.Tensor) -> None:
    r"""Ensure correct user provided correct disparity torch.Tensor.

    Args:
        disparity_tensor: The disparity torch.Tensor of shape :math:`(B, H, W, 1)`.

    """
    if not isinstance(disparity_tensor, torch.Tensor):
        raise StereoException(
            f"Expected 'disparity_tensor' to be an instance of torch.Tensor but got {type(disparity_tensor)}."
        )

    if len(disparity_tensor.shape) != 4:
        raise StereoException(f"Expected 'disparity_tensor' to have 4 dimensions. Got {disparity_tensor.shape}.")

    if disparity_tensor.shape[-1] != 1:
        raise StereoException(
            "Expected 'disparity_tensor' to have channels-last shape (B, H, W, 1) "
            "with a single channel in the last dimension. "
            f"Got {disparity_tensor.shape}."
        )

    if disparity_tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise StereoException(
            "Expected 'disparity_tensor' to have dtype torch.bfloat16, torch.float16, torch.float32 or torch.float64."
            f"Got {disparity_tensor.dtype}"
        )


def _check_Q_matrix(Q_matrix: torch.Tensor) -> None:
    r"""Ensure Q matrix is of correct form.

    Args:
        Q_matrix: The Q matrix for reprojecting disparity to a point cloud of shape :math:`(B, 4, 4)`

    """
    if not isinstance(Q_matrix, torch.Tensor):
        raise StereoException(f"Expected 'Q_matrix' to be an instance of torch.Tensor but got {type(Q_matrix)}.")

    if not len(Q_matrix.shape) == 3:
        raise StereoException(f"Expected 'Q_matrix' to have 3 dimensions. Got {Q_matrix.shape}")

    if not Q_matrix.shape[1:] == (4, 4):
        raise StereoException(f"Expected last two dimensions of 'Q_matrix' to be of shape (4, 4). Got {Q_matrix.shape}")

    if Q_matrix.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise StereoException(
            "Expected 'Q_matrix' to be of type torch.bfloat16, torch.float16, torch.float32 or torch.float64. "
            f"Got {Q_matrix.dtype}"
        )


def reproject_disparity_to_3D(disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor) -> torch.Tensor:
    r"""Reproject the disparity torch.Tensor to a 3D point cloud.

    See the Convention block on :class:`~kornia.geometry.camera.stereo.StereoCamera`.

    Args:
        disparity_tensor: Disparity torch.Tensor of shape :math:`(B, H, W, 1)`.
        Q_matrix: torch.Tensor of Q matrices of shapes :math:`(B, 4, 4)`.

    Returns:
        The 3D point cloud of shape :math:`(B, H, W, 3)`

    """
    _check_Q_matrix(Q_matrix)
    _check_disparity_tensor(disparity_tensor)

    batch_size, rows, cols, _ = disparity_tensor.shape
    dtype = disparity_tensor.dtype
    device = disparity_tensor.device

    uv = create_meshgrid(rows, cols, normalized_coordinates=False, device=device, dtype=dtype)
    uv = uv.expand(batch_size, -1, -1, -1)
    # create_meshgrid(normalized_coordinates=False) returns (x, y), so uv[..., 0] is the
    # column and uv[..., 1] is the row. u is the column and v the row, as in
    # cv2.reprojectImageTo3D, whose semantics this function provides (#2042). Unbinding
    # them the other way round fed the row into u and the column into v, which transposed
    # the two pixel indices in the result.
    u, v = torch.unbind(uv, dim=-1)
    u, v = torch.unsqueeze(u, -1), torch.unsqueeze(v, -1)
    uvd = torch.stack((u, v, disparity_tensor), 1).reshape(batch_size, 3, -1).permute(0, 2, 1)
    points = transform_points(Q_matrix, uvd).reshape(batch_size, rows, cols, 3)

    # Final check that everything went well.
    if not points.shape == (batch_size, rows, cols, 3):
        raise StereoException(
            "Something went wrong in `reproject_disparity_to_3D`. Expected the final output"
            f"to be of shape {(batch_size, rows, cols, 3)}."
            f"But the computed point cloud had shape {points.shape}. "
            "Please ensure input are correct. If this is an error, please submit an issue."
        )
    return points
