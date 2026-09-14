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
        - the baseline is read back from that column as ``tx = -P_right[0, 3] / fx``, and :attr:`Q` is built
          from ``fx``, ``fy``, ``cx_left``, ``cy`` and that ``tx``. Note which focal length sits in which row:
          ``Q[0, 0]`` carries ``fy`` and ``Q[1, 1]`` carries ``fx``, while the homogeneous divide is by
          ``-fy * disparity``, so the two cancel and the first output coordinate ends up scaled by ``1 / fx``
          and the second by ``1 / fy``.
        - a point cloud is a homogeneous transform by :attr:`Q` followed by the divide by ``W``. When
          ``abs(W) > 1e-8``, an overall sign on :attr:`Q` cancels, so :attr:`Q` and its negation return the
          same points. For ``abs(W) <= 1e-8``, the homogeneous conversion returns the numerator unchanged,
          and the two matrices return opposing values. :attr:`Q` is exactly the matrix written out on the
          :doc:`/geometry.camera.stereo` page, above this docstring, evaluated at the page's own ``tx``: the
          page's :math:`P_1` carries ``fx * tx`` in its last column, so that ``tx`` is ``P_right[0, 3] / fx``,
          which the constructor rejects only when it is **positive** (``tx = 0`` and a batch with one positive
          product pass, see the second warning below). The :attr:`tx` attribute exposes the negation of that
          symbol, ``-P_right[0, 3] / fx``; substituting the attribute's value for the page's ``tx`` gives
          neither :attr:`Q` nor its negation, because the page's last row carries no ``tx`` and does not flip.
        - a disparity map is channels-**last**, :math:`(B, H, W, 1)`, for
          :meth:`~kornia.geometry.camera.stereo.StereoCamera.reproject_disparity_to_3D` and for the module-level
          :func:`~kornia.geometry.camera.stereo.reproject_disparity_to_3D` alike -- the :math:`(B, 1, H, W)`
          layout the rest of kornia uses for images is rejected -- and the returned point cloud is
          :math:`(B, H, W, 3)`.
        - the pixels are the integer pixel centres that :func:`~kornia.geometry.grid.create_meshgrid`
          enumerates, described in the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - ``u`` is the **column** index and ``v`` the **row** index, as in ``cv2.reprojectImageTo3D``:
          :math:`X = (u - c_x) Z / f_x` and :math:`Y = (v - c_y) Z / f_y`.

    .. warning::
        Several of the constructor guards do not enforce the contract above. A differing ``cx`` is
        **rejected**, even though :attr:`cx_left` and :attr:`cx_right` are exposed separately and
        ``Q[3, 3]`` carries ``fy * (cx_left - cx_right)`` for exactly that case, so that factor is zero on
        any rig the constructor accepts. The ``tx * fx < 0`` guard is quantified with ``torch.all``, so a
        batch whose second element has the two cameras the wrong way round is accepted and reprojects that
        element behind the camera. And ``tx = 0`` passes the same guard, collapsing ``Q`` so that every
        disparity reprojects to the origin with no ``inf`` to notice. These guard issues are tracked as
        `#4270 <https://github.com/kornia/kornia/issues/4270>`_ and pinned by
        ``test_wart_stereo_rejects_differing_principal_points_4270``,
        ``test_wart_stereo_accepts_a_batch_with_one_positive_tx_fx_4270``,
        ``test_wart_stereo_tx_zero_collapses_every_point_to_the_origin_4270`` in
        ``tests/geometry/camera/test_stereo.py``.

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

        # Ensure that tx * fx is negative and exists.
        tx_fx = rectified_right_camera[..., 0, 3]
        if not is_exporting() and tx_fx.numel() > 0 and torch.all(torch.gt(tx_fx, 0)):
            raise StereoException(f"Expected :math:`T_x * f_x` to be negative. Got {tx_fx}.")

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
