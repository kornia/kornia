from typing import Any

import torch

from kornia.core import Tensor, stack, zeros
from kornia.geometry.linalg import transform_points
from kornia.utils.grid import create_meshgrid


class StereoException(Exception):
    def __init__(self, msg: str, *args: Any, **kwargs: Any) -> None:
        r"""Custom exception for the :module:`~kornia.geometry.camera.stereo` module.

        Adds a general helper module redirecting the user to the proper documentation site.

        Args:
            msg: Custom message to add to the general message.
            *args:
            **kwargs:
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
    def __init__(self, rectified_left_camera: Tensor, rectified_right_camera: Tensor) -> None:
        r"""Class representing a horizontal stereo camera setup.

        Args:
            rectified_left_camera: The rectified left camera projection matrix
              of shape :math:`(B, 3, 4)`
            rectified_right_camera: The rectified right camera projection matrix
              of shape :math:`(B, 3, 4)`
        """
        self._check_stereo_camera(rectified_left_camera, rectified_right_camera)
        self.rectified_left_camera: Tensor = rectified_left_camera
        self.rectified_right_camera: Tensor = rectified_right_camera

        self.device = self.rectified_left_camera.device
        self.dtype = self.rectified_left_camera.dtype

        self._Q_matrix = self._init_Q_matrix()

    @staticmethod
    def _check_stereo_camera(rectified_left_camera: Tensor, rectified_right_camera: Tensor) -> None:
        r"""Utility function to ensure user specified correct camera matrices.

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

        if rectified_left_camera.shape[:1] == (3, 4):
            raise StereoException(
                f"Expected each 'rectified_left_camera' to be of shape (3, 4).Got {rectified_left_camera.shape[:1]}."
            )

        if rectified_right_camera.shape[:1] == (3, 4):
            raise StereoException(
                f"Expected each 'rectified_right_camera' to be of shape (3, 4).Got {rectified_right_camera.shape[:1]}."
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
        if not torch.all(torch.eq(rectified_left_camera[..., :, :3], rectified_right_camera[..., :, :3])):
            raise StereoException(
                "Expected 'left_rectified_camera' and 'rectified_right_camera' to have"
                "same parameters except for the last column."
                f"Got {rectified_left_camera[..., :, :3]} and {rectified_right_camera[..., :, :3]}."
            )

        # Ensure that tx * fx is negative and exists.
        tx_fx = rectified_right_camera[..., 0, 3]
        if torch.all(torch.gt(tx_fx, 0)):
            raise StereoException(f"Expected :math:`T_x * f_x` to be negative. Got {tx_fx}.")

    @property
    def batch_size(self) -> int:
        r"""Return the batch size of the storage.

        Returns:
           scalar with the batch size
        """
        return self.rectified_left_camera.shape[0]

    @property
    def fx(self) -> Tensor:
        r"""Return the focal length in the x-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> Tensor:
        r"""Returns the focal length in the y-direction.

        Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx_left(self) -> Tensor:
        r"""Return the x-coordinate of the principal point for the left camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cx_right(self) -> Tensor:
        r"""Return the x-coordinate of the principal point for the right camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_right_camera[..., 0, 2]

    @property
    def cy(self) -> Tensor:
        r"""Return the y-coordinate of the principal point.

        Note that the y-coordinate of the principal points
        is assumed to be equal for the left and right camera.

        Returns:
            tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> Tensor:
        r"""The horizontal baseline between the two cameras.

        Returns:
            Tensor of shape :math:`(B)`
        """
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> Tensor:
        r"""The Q matrix of the horizontal stereo setup.

        This matrix is used for reprojecting a disparity tensor to
        the corresponding point cloud. Note that this is in a general form that allows different focal
        lengths in the x and y direction.

        Return:
            The Q matrix of shape :math:`(B, 4, 4)`.
        """
        return self._Q_matrix

    def _init_Q_matrix(self) -> Tensor:
        r"""Initialized the Q matrix of the horizontal stereo setup. See the Q property.

        Returns:
            The Q matrix of shape :math:`(B, 4, 4)`.
        """
        Q = zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        baseline: Tensor = -self.tx
        Q[:, 0, 0] = self.fy * baseline
        Q[:, 0, 3] = -self.fy * self.cx_left * baseline
        Q[:, 1, 1] = self.fx * baseline
        Q[:, 1, 3] = -self.fx * self.cy * baseline
        Q[:, 2, 3] = self.fx * self.fy * baseline
        Q[:, 3, 2] = -self.fy
        Q[:, 3, 3] = self.fy * (self.cx_left - self.cx_right)  # NOTE: This is usually zero.
        return Q

    def reproject_disparity_to_3D(self, disparity_tensor: Tensor) -> Tensor:
        r"""Reproject the disparity tensor to a 3D point cloud.

        Args:
            disparity_tensor: Disparity tensor of shape :math:`(B, 1, H, W)`.

        Returns:
            The 3D point cloud of shape :math:`(B, H, W, 3)`
        """
        return reproject_disparity_to_3D(disparity_tensor, self.Q)


def _check_disparity_tensor(disparity_tensor: Tensor) -> None:
    r"""Utility function to ensure correct user provided correct disparity tensor.

    Args:
        disparity_tensor: The disparity tensor of shape :math:`(B, 1, H, W)`.
    """
    if not isinstance(disparity_tensor, Tensor):
        raise StereoException(
            f"Expected 'disparity_tensor' to be an instance of Tensor but got {type(disparity_tensor)}."
        )

    if len(disparity_tensor.shape) != 4:
        raise StereoException(f"Expected 'disparity_tensor' to have 4 dimensions. Got {disparity_tensor.shape}.")

    if disparity_tensor.shape[-1] != 1:
        raise StereoException(
            "Expected dimension 1 of 'disparity_tensor' to be 1 for as single channeled disparity map."
            f"Got {disparity_tensor.shape}."
        )

    if disparity_tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        raise StereoException(
            "Expected 'disparity_tensor' to have dtype torch.float16, torch.float32 or torch.float64."
            f"Got {disparity_tensor.dtype}"
        )


def _check_Q_matrix(Q_matrix: Tensor) -> None:
    r"""Utility function to ensure Q matrix is of correct form.

    Args:
        Q_matrix: The Q matrix for reprojecting disparity to a point cloud of shape :math:`(B, 4, 4)`
    """
    if not isinstance(Q_matrix, Tensor):
        raise StereoException(f"Expected 'Q_matrix' to be an instance of Tensor but got {type(Q_matrix)}.")

    if not len(Q_matrix.shape) == 3:
        raise StereoException(f"Expected 'Q_matrix' to have 3 dimensions. Got {Q_matrix.shape}")

    if not Q_matrix.shape[1:] == (4, 4):
        raise StereoException(f"Expected last two dimensions of 'Q_matrix' to be of shape (4, 4). Got {Q_matrix.shape}")

    if Q_matrix.dtype not in (torch.float16, torch.float32, torch.float64):
        raise StereoException(
            f"Expected 'Q_matrix' to be of type torch.float16, torch.float32 or torch.float64. Got {Q_matrix.dtype}"
        )


def reproject_disparity_to_3D(disparity_tensor: Tensor, Q_matrix: Tensor) -> Tensor:
    r"""Reproject the disparity tensor to a 3D point cloud.

    Args:
        disparity_tensor: Disparity tensor of shape :math:`(B, H, W, 1)`.
        Q_matrix: Tensor of Q matrices of shapes :math:`(B, 4, 4)`.

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
    v, u = torch.unbind(uv, dim=-1)
    v, u = torch.unsqueeze(v, -1), torch.unsqueeze(u, -1)
    uvd = stack((u, v, disparity_tensor), 1).reshape(batch_size, 3, -1).permute(0, 2, 1)
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
