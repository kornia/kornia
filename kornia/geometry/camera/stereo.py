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

"""Rectified stereo: reproject disparity maps to 3D points, from the rig's projection matrices or from a Q matrix.

A disparity map is channels-last, ``(B, H, W, 1)``, and holds ``d = u_left - u_right`` in pixels, positive for a
point in front of the rig. Pixel ``(u, v)`` is (column, row) with integer pixel centres, as in
``cv2.reprojectImageTo3D``, and the points are ``(B, H, W, 3)``.

Which function to use:

- A calibrated rig given by its two rectified ``(B, 3, 4)`` projection matrices (``P1`` and ``P2`` of
  ``cv2.stereoRectify``): :class:`StereoCamera`, whose
  :meth:`~kornia.geometry.camera.stereo.StereoCamera.reproject_disparity_to_3D` returns points in the left
  rectified camera frame.
- A ``(B, 4, 4)`` ``Q`` matrix from elsewhere: :func:`reproject_disparity_to_3D`.
- The error both raise for an invalid rig, disparity map or ``Q``: :class:`StereoException`.
- A depth map instead of points: :func:`~kornia.geometry.depth.depth_from_disparity`.
"""

from typing import Any, Literal, overload

import torch

from kornia.core.utils import is_compiling
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.grid import create_meshgrid

__all__ = ["StereoCamera", "StereoException", "reproject_disparity_to_3D"]


class StereoException(Exception):
    """Signal an invalid stereo rig, disparity map or Q matrix, with a link to the stereo docs.

    ``raise StereoException(msg)`` raises an exception whose message is ``msg`` followed by a link to the
    :doc:`/geometry.camera.stereo` page. :class:`StereoCamera` raises it for projection matrices of the wrong
    shape, on different devices or with different dtypes and, outside graph capture, for different intrinsics or a
    baseline that is zero or points the wrong way; :func:`reproject_disparity_to_3D` raises it for a disparity map
    or ``Q`` of the wrong type, shape or dtype. It derives from :class:`Exception` directly, so
    ``except ValueError`` does not catch it; a ``Q`` batch that is neither 1 nor the disparity batch raises
    :class:`ValueError` instead.

    See Also:
        - :class:`StereoCamera`: the rig whose constructor checks raise it.
        - :func:`reproject_disparity_to_3D`: the function whose input checks raise it.

    """

    def __init__(self, msg: str, *args: Any, **kwargs: Any) -> None:
        r"""Construct custom exception for the :mod:`kornia.geometry.camera.stereo` module.

        Appends a link to the stereo documentation page to ``msg``.

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
    """Model a rectified horizontal stereo rig and reproject its disparity maps to 3D points.

    ``stereo = StereoCamera(P_left, P_right)`` takes the ``(B, 3, 4)`` rectified projection matrices of the left and
    the right camera, such as ``P1`` and ``P2`` of ``cv2.stereoRectify``, and
    ``points = stereo.reproject_disparity_to_3D(disparity)`` takes a channels-last ``(B, H, W, 1)`` disparity map
    ``d = u_left - u_right``, in pixels, and returns the 3D point of every pixel in the left rectified camera
    frame, ``(B, H, W, 3)``, in the units of the baseline: ``Z = fx * tx / d``. For nonzero disparities the result
    matches ``cv2.reprojectImageTo3D`` with OpenCV's ``Q``. With a ``Q`` matrix from elsewhere, call
    :func:`reproject_disparity_to_3D` instead; for a depth map rather than points,
    :func:`~kornia.geometry.depth.depth_from_disparity`.

    Convention:
        - the two arguments are the **rectified projection matrices** of the left and the right camera, each of
          shape ``(B, 3, 4)``: ``[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]`` for the left camera, and
          the same matrix with ``-tx * fx`` in the last column for the right one. The constructor requires the
          two to be equal outside that last column.
        - the baseline is read back as the attribute :attr:`tx` ``= -P_right[0, 3] / fx``, which must be
          strictly **positive** for every rig in the batch: a zero baseline and swapped cameras both raise. A
          non-finite ``P_right[0, 3]`` is not screened, and these checks and the equal-intrinsics check are
          skipped under graph capture (``torch.compile``, ``torch.export``).
          The :doc:`/geometry.camera.stereo` page's symbol ``tx`` is the negation of this attribute; :attr:`Q`
          is the page's matrix evaluated at the page's ``tx``.
        - ``Q[0, 0]`` carries ``fy`` and ``Q[1, 1]`` carries ``fx``; with the divide by ``W = -fy * disparity``
          the result is ``X = (u - cx) Z / fx``, ``Y = (v - cy) Z / fy``, ``Z = fx * tx / disparity``.
        - ``u`` is the **column** index and ``v`` the **row** index, as in ``cv2.reprojectImageTo3D``.
        - a disparity map is channels-**last**, ``(B, H, W, 1)`` (a ``(B, 1, H, W)`` map is rejected),
          and the point cloud is ``(B, H, W, 3)``. A positive disparity puts the point in front of the rig and a
          negative one behind it. A rig with ``B = 1`` serves a batch of disparity maps; any other batch must
          match the disparity batch, else :class:`ValueError`.
        - pixel centres are integers; see the Convention block on
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - the constructor raises :class:`StereoException` for an invalid rig, and the method for a disparity map
          that is not a floating ``(B, H, W, 1)`` tensor. An integer rig is accepted by the constructor, but its
          ``Q`` is integer too, so every call of the method then raises
          (`#6471 <https://github.com/kornia/kornia/issues/6471>`_).

    .. warning::
        A differing ``cx`` is **rejected**, although ``Q[3, 3] = fy * (cx_left - cx_right)`` exists for that
        case, so it is zero on every rig the constructor accepts outside graph capture:
        `#4270 <https://github.com/kornia/kornia/issues/4270>`_. A zero disparity (a point at infinity) makes
        ``W = 0``; the divide is then skipped and a finite placeholder, behind the camera on a real rig, is
        returned. Request ``return_mask=True`` from :meth:`reproject_disparity_to_3D` to identify numerical
        reconstruction failures without changing those coordinates or their gradients.

    Args:
        rectified_left_camera: The rectified left camera projection matrix of shape ``(B, 3, 4)``.
        rectified_right_camera: The rectified right camera projection matrix of shape ``(B, 3, 4)``.

    See Also:
        - :func:`reproject_disparity_to_3D`: the same reprojection from a ``Q`` matrix instead of the two
          projection matrices.
        - :func:`~kornia.geometry.depth.depth_from_disparity`: a depth map, ``fx * tx / d`` at every pixel,
          instead of points.
        - :func:`~kornia.geometry.depth.depth_to_3d_v2`: points from a depth map; after
          :func:`~kornia.geometry.depth.depth_from_disparity`, the same points up to the ``1e-8`` that function
          adds to the disparity.
        - :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: a single camera with intrinsics and
          world-to-camera extrinsics.
    """

    def __init__(self, rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor) -> None:
        r"""Class representing a horizontal stereo camera setup.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape ``(B, 3, 4)``
            rectified_right_camera: The rectified right camera projection matrix
              of shape ``(B, 3, 4)``

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
              of shape ``(B, 3, 4)``
            rectified_right_camera: The rectified right camera projection matrix
              of shape ``(B, 3, 4)``

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
        # The check reads the data, which graph capture cannot do; skip it under any capture (torch.compile,
        # torch.export).
        if not is_compiling() and not torch.all(
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
        # The check reads the data, which graph capture cannot do; skip it under any capture (torch.compile,
        # torch.export).
        tx_fx = rectified_right_camera[..., 0, 3]
        if not is_compiling():
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
            torch.Tensor of shape ``(B)``

        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        r"""Returns the focal length in the y-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            torch.Tensor of shape ``(B)``

        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx_left(self) -> torch.Tensor:
        r"""Return the x-coordinate of the principal point for the left camera.

        Returns:
            torch.Tensor of shape ``(B)``

        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cx_right(self) -> torch.Tensor:
        r"""Return the x-coordinate of the principal point for the right camera.

        Returns:
            torch.Tensor of shape ``(B)``

        """
        return self.rectified_right_camera[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        r"""Return the y-coordinate of the principal point.

        Note that the y-coordinate of the principal points
        is assumed to be equal for the left and right camera.

        Returns:
            torch.Tensor of shape ``(B)``

        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> torch.Tensor:
        r"""The horizontal baseline between the two cameras.

        Returns:
            torch.Tensor of shape ``(B)``

        """
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> torch.Tensor:
        r"""The Q matrix of the horizontal stereo setup.

        This matrix is used for reprojecting a disparity torch.Tensor to
        the corresponding point cloud. Note that this is in a general form that allows different focal
        lengths in the x and y direction.

        Return:
            The Q matrix of shape ``(B, 4, 4)``.

        """
        return self._Q_matrix

    def _init_Q_matrix(self) -> torch.Tensor:
        r"""Initialize the Q matrix of the horizontal stereo setup. See the Q property.

        Returns:
            The Q matrix of shape ``(B, 4, 4)``.

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

    @overload
    def reproject_disparity_to_3D(
        self, disparity_tensor: torch.Tensor, return_mask: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def reproject_disparity_to_3D(
        self, disparity_tensor: torch.Tensor, return_mask: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def reproject_disparity_to_3D(
        self, disparity_tensor: torch.Tensor, return_mask: bool
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    def reproject_disparity_to_3D(
        self, disparity_tensor: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Reproject the disparity torch.Tensor to a 3D point cloud.

        See the Convention block on :class:`~kornia.geometry.camera.stereo.StereoCamera`.

        Args:
            disparity_tensor: Disparity torch.Tensor of shape ``(B, H, W, 1)``.
            return_mask: If True, also return numerical reconstruction validity with
                shape ``(B, H, W, 1)``. This only adds diagnostics; it does not fill
                invalid points or change their gradients. See the module-level
                :func:`reproject_disparity_to_3D` for the mask's complete contract.

        Returns:
            The 3D point cloud of shape ``(B, H, W, 3)``, in the left rectified
            camera frame, or ``(points, valid)`` when ``return_mask=True``.

        """
        return reproject_disparity_to_3D(disparity_tensor, self.Q, return_mask=return_mask)


def _check_disparity_tensor(disparity_tensor: torch.Tensor) -> None:
    r"""Ensure correct user provided correct disparity torch.Tensor.

    Args:
        disparity_tensor: The disparity torch.Tensor of shape ``(B, H, W, 1)``.

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
        Q_matrix: The Q matrix for reprojecting disparity to a point cloud of shape ``(B, 4, 4)``

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


@overload
def reproject_disparity_to_3D(
    disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor, return_mask: Literal[False] = False
) -> torch.Tensor: ...


@overload
def reproject_disparity_to_3D(
    disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor, return_mask: Literal[True]
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def reproject_disparity_to_3D(
    disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor, return_mask: bool
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def reproject_disparity_to_3D(
    disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor, return_mask: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Reproject a disparity map to 3D points with a Q matrix, as cv2.reprojectImageTo3D does.

    ``points = reproject_disparity_to_3D(disparity, Q)`` takes a channels-last ``(B, H, W, 1)`` disparity map and
    ``Q`` matrices ``(B, 4, 4)`` or ``(1, 4, 4)``, and returns ``(X / W, Y / W, Z / W)`` of
    ``Q (u, v, d, 1)`` for every pixel ``(u, v)`` = (column, row), ``(B, H, W, 3)``, in the frame ``Q`` defines.
    Where ``abs(W) <= 1e-8``, as at a zero disparity when ``Q[3, 3] = 0``, it returns the undivided ``(X, Y, Z)``,
    while OpenCV divides (infinities at ``W = 0``); see the Note. With ``return_mask=True`` it also returns a
    ``(B, H, W, 1)`` bool mask of numerically valid points. For a rig given by its two rectified projection
    matrices, :class:`StereoCamera` builds ``Q``; for a depth map rather than points, use
    :func:`~kornia.geometry.depth.depth_from_disparity`.

    See the Convention block on :class:`~kornia.geometry.camera.stereo.StereoCamera`.

    Args:
        disparity_tensor: Disparity torch.Tensor of shape ``(B, H, W, 1)``.
        Q_matrix: torch.Tensor of Q matrices of shape ``(B, 4, 4)`` or ``(1, 4, 4)``.
        return_mask: If True, also return a boolean numerical reconstruction mask of
            shape ``(B, H, W, 1)``, on the input device. Defaults to False.

    Returns:
        The 3D point cloud of shape ``(B, H, W, 3)``, or ``(points, valid)`` when
        ``return_mask=True``. Points retain the disparity dtype in either mode.

    Raises:
        ValueError: If the Q matrix batch size is neither 1 nor the disparity batch size.
        StereoException: If ``disparity_tensor`` or ``Q_matrix`` is not a floating tensor of the documented shape.

    See Also:
        - :class:`StereoCamera`: builds ``Q`` from the two rectified projection matrices of a rig.
        - :func:`~kornia.geometry.depth.depth_from_disparity`: a depth map instead of points, from a baseline and a
          focal length.
        - :func:`~kornia.geometry.depth.depth_to_3d_v2`: points from a depth map and a ``3x3`` ``K``.

    Note:
        A point is valid when all four homogeneous components are finite,
        ``abs(W) > 1e-8`` in the converter's working dtype, and XYZ is finite after
        conversion to the returned disparity dtype. The matrix multiplication and
        homogeneous conversion use Q's dtype (subject to the active autocast context).
        The absolute W cutoff is the existing coordinate policy: scaling Q can change
        validity near the cutoff, while Q and -Q have the same mask.

        This mask does not certify a stereo match, front-of-camera eligibility,
        image bounds or occlusion. In particular, zero disparity can reconstruct a
        finite point when Q has a nonzero offset. Combine the mask with the matcher's
        own validity information. The output frame is defined by Q; for a
        :class:`StereoCamera` it is the left rectified camera frame.

        Requesting the mask changes neither coordinates nor gradients. Below the W
        cutoff, the converter still returns the homogeneous numerator as a placeholder;
        nonfinite inputs can still produce nonfinite coordinates or gradients. This is
        a diagnostic API, not a safe invalid-point filling policy. Masking a loss after
        unsafe arithmetic is not guaranteed to prevent nonfinite gradients.

    """
    _check_Q_matrix(Q_matrix)
    _check_disparity_tensor(disparity_tensor)

    batch_size, rows, cols, _ = disparity_tensor.shape
    dtype = disparity_tensor.dtype
    device = disparity_tensor.device
    if Q_matrix.shape[0] != batch_size and Q_matrix.shape[0] != 1:
        raise ValueError("Q_matrix batch size must match disparity_tensor or be 1.")

    uv = create_meshgrid(rows, cols, normalized_coordinates=False, device=device, dtype=dtype)
    uv = uv.expand(batch_size, -1, -1, -1)
    # create_meshgrid(normalized_coordinates=False) returns (x, y), so uv[..., 0] is the
    # column and uv[..., 1] is the row. u is the column and v the row, as in
    # cv2.reprojectImageTo3D, whose semantics this function provides (#2042). Unbinding
    # them the other way round fed the row into u and the column into v, which transposed
    # the two pixel indices in the result.
    u, v = torch.unbind(uv, dim=-1)
    u, v = torch.unsqueeze(u, -1), torch.unsqueeze(v, -1)
    # The pixel count is spelled out: a ``-1`` cannot be inferred for an empty batch.
    uvd = torch.stack((u, v, disparity_tensor), 1).reshape(batch_size, 3, rows * cols).permute(0, 2, 1)
    # Preserve transform_points' empty point-set path, including its lack of a Q
    # gradient connection. There are no entries to diagnose in the empty mask.
    if rows * cols == 0:
        points = uvd.reshape(batch_size, rows, cols, 3)
        if return_mask:
            return points, torch.empty_like(disparity_tensor, dtype=torch.bool)
        return points
    # Keep the actual homogeneous components for diagnostics: finite output alone
    # cannot distinguish a placeholder, or finite XYZ produced by infinite W.
    # This is transform_points' existing bmm/conversion/cast path, shared by both
    # return modes so requesting diagnostics cannot select different arithmetic.
    homogeneous = torch.bmm(
        convert_points_to_homogeneous(uvd).to(Q_matrix.dtype),
        Q_matrix.expand(batch_size, -1, -1).transpose(-1, -2),
    )
    eps = 1e-8
    points = convert_points_from_homogeneous(homogeneous, eps=eps).to(dtype).reshape(batch_size, rows, cols, 3)

    # Final check that everything went well.
    if not points.shape == (batch_size, rows, cols, 3):
        raise StereoException(
            "Something went wrong in `reproject_disparity_to_3D`. Expected the final output"
            f"to be of shape {(batch_size, rows, cols, 3)}."
            f"But the computed point cloud had shape {points.shape}. "
            "Please ensure input are correct. If this is an error, please submit an issue."
        )
    if return_mask:
        valid = torch.isfinite(homogeneous).all(-1, keepdim=True) & (homogeneous[..., -1:].abs() > eps)
        valid = valid.reshape(batch_size, rows, cols, 1) & torch.isfinite(points).all(-1, keepdim=True)
        return points, valid
    return points
