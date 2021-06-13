import torch


class StereoException(Exception):
    def __init__(self, msg, *args, **kwargs):
        """
        Custom exception for the :module:`~kornia.geometry.camera.stereo` module. Adds a general helper module
        redirecting the user to the proper documentation site.
        Args:
            msg: Custom message to add to the general message.
            *args:
            **kwargs:
        """
        doc_help = "\n Please check documents here: " \
                   "https://kornia.readthedocs.io/en/latest/geometry.camera.stereo.html" \
                   "for further information and examples."
        final_msg = msg + doc_help
        super().__init__(final_msg, *args, **kwargs)  # type: ignore


class StereoCamera:
    def __init__(self, rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor):
        """
        Class representing a horizontal stereo camera setup.

        Args:
            rectified_left_camera (torch.Tensor): The rectified left camera projection matrix
                                                    of shape :math:`(B, 3, 4)`
            rectified_right_camera (torch.Tensor): The rectified right camera projection matrix
                                                    of shape :math:`(B, 3, 4)`
        """
        self._check_stereo_camera(rectified_left_camera, rectified_right_camera)
        self.rectified_left_camera: torch.Tensor = rectified_left_camera
        self.rectified_right_camera: torch.Tensor = rectified_right_camera

        self.device: torch.device = self.rectified_left_camera.device
        self.dtype: torch.dtype = self.rectified_left_camera.dtype

    @staticmethod
    def _check_stereo_camera(rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor):
        """
        Utility function to ensure user specified correct camera matrices.
        Args:
            rectified_left_camera (torch.Tensor): The rectified left camera projection matrix
                                                    of shape :math:`(B, 3, 4)`
            rectified_right_camera (torch.Tensor): The rectified right camera projection matrix
                                                    of shape :math:`(B, 3, 4)`
        """
        # Ensure correct shapes
        if rectified_left_camera.ndim != 3:
            raise StereoException(f"Expected 'rectified_left_camera' to have 3 dimensions. "
                                  f"Got {rectified_left_camera.ndim}.")

        if rectified_right_camera.ndim != 3:
            raise StereoException(f"Expected 'rectified_right_camera' to have 3 dimension. "
                                  f"Got {rectified_right_camera.ndim}.")

        if rectified_left_camera.shape[:1] == (3, 4):
            raise StereoException(f"Expected each 'rectified_left_camera' to be of shape (3, 4)."
                                  f"Got {rectified_left_camera.shape[:1]}.")

        if rectified_right_camera.shape[:1] == (3, 4):
            raise StereoException(f"Expected each 'rectified_right_camera' to be of shape (3, 4)."
                                  f"Got {rectified_right_camera.shape[:1]}.")

        # Ensure same devices for cameras.
        if rectified_left_camera.device != rectified_right_camera.device:
            raise StereoException(f"Expected 'rectified_left_camera' and 'rectified_right_camera' "
                                  f"to be on the same devices."
                                  f"Got {rectified_left_camera.device} and {rectified_right_camera.device}.")

        # Ensure same dtypes for cameras.
        if rectified_left_camera.dtype != rectified_right_camera.dtype:
            raise StereoException(f"Expected 'rectified_left_camera' and 'rectified_right_camera' to"
                                  f"have same dtype."
                                  f"Got {rectified_left_camera.dtype} and {rectified_right_camera.dtype}.")

        # Ensure all intrinsics parameters (fx, fy, cx, cy) are the same in both cameras.
        if not torch.all(torch.eq(rectified_left_camera[..., :, :3], rectified_right_camera[..., :, :3])):
            raise StereoException(f"Expected 'left_rectified_camera' and 'rectified_right_camera' to have"
                                  f"same parameters except for the last column."
                                  f"Got {rectified_left_camera[..., :, :3]} and {rectified_right_camera[..., :, :3]}.")

        # Ensure that tx * fx is negative and exists.
        tx_fx = rectified_right_camera[..., 0, 3]
        if torch.all(torch.gt(tx_fx, 0)):
            raise StereoException(f"Expected :math:`T_x * f_x` to be negative."
                                  f"Got {tx_fx}.")

    @property
    def batch_size(self) -> int:
        r"""Returns the batch size of the storage.

        Returns:
            int: scalar with the batch size
        """
        return self.rectified_left_camera.shape[0]

    @property
    def fx(self) -> torch.Tensor:
        r"""Returns the focal length in the x-direction. Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        r"""Returns the focal length in the y-direction. Note that the focal lengths of the rectified left and right
        camera are assumed to be equal.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx_left(self) -> torch.Tensor:
        r"""Returns the x-coordinate of the principal point for the left camera.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cx_right(self) -> torch.Tensor:
        r"""Returns the x-coordinate of the principal point for the right camera.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_right_camera[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        r"""Returns the y-coordinate of the principal point. Note that the y-coordinate of the principal points
        is assumed to be equal for the left and right camera.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> torch.Tensor:
        """
        The horizontal baseline between the two cameras.

        Returns:
            torch.tensor: Tensor of shape :math:`(B)`

        """
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> torch.Tensor:
        """
        The Q matrix of the horizontal stereo setup. This matrix is used for reprojecting a disparity tensor to
        the corresponding point cloud. Note that this is in a general form that allows different focal
        lengths in the x and y direction.

        Return:
            torch.Tensor: The Q matrix of shape :math:`(B, 4, 4)`.
        """
        Q = torch.zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        baseline = -self.tx
        Q[:, 0, 0] = self.fy * baseline
        Q[:, 0, 3] = -self.fy * self.cx_left * baseline
        Q[:, 1, 1] = self.fx * baseline
        Q[:, 1, 3] = -self.fx * self.cy * baseline
        Q[:, 2, 3] = self.fx * self.fy * baseline
        Q[:, 3, 2] = -self.fy
        Q[:, 3, 3] = self.fy * (self.cx_left - self.cx_right)  # NOTE: This is usually zero.
        return Q

    def reproject_disparity_to_3D(self, disparity_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reproject the disparity tensor to a 3D point cloud.

        Args:
            disparity_tensor (torch.Tensor): Disparity tensor of shape :math:`(B, H, W)`.

        Returns:
            torch.Tensor: The 3D point cloud of shape :math:`(B, H * W, 3)`
        """
        return reproject_disparity_to_3D(disparity_tensor, self.Q)


def _check_disparity_tensor(disparity_tensor: torch.Tensor):
    """
    Utility function to ensure correct user provided correct disparity tensor.
    Args:
        disparity_tensor (torch.Tensor): The disparity tensor of shape :math:`(B, H, W)`.
    """
    if disparity_tensor.ndim != 3:
        raise StereoException(f"Expected 'disparity_tensor' to have 3 dimensions."
                              f"Got {disparity_tensor.ndim}.")

    if disparity_tensor.dtype != torch.float32:
        raise StereoException(f"Expected 'disparity_tensor' to have dtype torch.float32."
                              f"Got {disparity_tensor.dtype}")


def _check_Q_matrix(Q_matrix: torch.Tensor):
    """
    Utility function to ensure Q matrix is of correct form.
    Args:
        Q_matrix (torch.Tensor): The Q matrix for reprojecting disparity to a point cloud of shape :math:`(B, 4, 4)`
    """
    if not Q_matrix.ndim == 3:
        raise StereoException(f"Expected 'Q_matrix to have 3 dimenstions."
                              f"Got {Q_matrix.ndim}")

    if not Q_matrix.shape[1:] == (4, 4):
        raise StereoException(f"Expected last two dimensions of 'Q_matrix' to be of shape (4, 4)."
                              f"Got {Q_matrix.shape}")

    if not Q_matrix.dtype == torch.float32:
        raise StereoException(f"Expected 'Q_matrix' to be of type torch.float32."
                              f"Got {Q_matrix.dtype}")


def reproject_disparity_to_3D(disparity_tensor: torch.Tensor, Q_matrix: torch.Tensor) -> torch.Tensor:
    """
    Reproject the disparity tensor to a 3D point cloud.

    Args:
        disparity_tensor (torch.Tensor): Disparity tensor of shape :math:`(B, H, W)`.
        Q_matrix (torch.Tensor): Tensor of Q matrices of shapes :math:`(B, 4, 4)`.

    Returns:
        torch.Tensor: The 3D point cloud of shape :math:`(B, H * W, 3)`
    """
    _check_Q_matrix(Q_matrix)
    _check_disparity_tensor(disparity_tensor)

    batch_size, rows, cols = disparity_tensor.shape
    dtype = disparity_tensor.dtype
    device = disparity_tensor.device
    homogenous_observation_ndim = 4
    euclidian_observation_ndim = homogenous_observation_ndim - 1

    # Construct a mesh grid of uv values, i.e. the tensors will contain 1:rows and 1:cols.
    u, v = torch.meshgrid(torch.arange(rows, dtype=dtype, device=device),
                          torch.arange(cols, dtype=dtype, device=device))
    u, v = u.expand(batch_size, -1, -1), v.expand(batch_size, -1, -1)

    # The z dimension in homogenous coordinates are just 1.
    z = torch.ones((batch_size, rows, cols), dtype=dtype, device=device)

    # Stack the observations into a tensor of shape (batch_size, 4, -1) that contains all
    # 4 dimensional vectors [u v disparity 1].
    uvdz = torch.stack((u, v, disparity_tensor, z), 1).reshape(batch_size, homogenous_observation_ndim, -1)

    # Matrix multiply all vectors with the Q matrix
    hom_points = torch.bmm(Q_matrix, uvdz)

    # Convert from homogenous to euclidian space.
    z_points = torch.unsqueeze(hom_points[:, euclidian_observation_ndim], 1)
    points = (hom_points / z_points)[:, :euclidian_observation_ndim]
    points = points.permute(0, 2, 1)

    # Final check that everything went well.
    if not points.shape == (batch_size, rows * cols, euclidian_observation_ndim):
        raise StereoException(f"Something went wrong in `reproject_disparity_to_3D`. Expected the final output"
                              f"to be of shape {(batch_size, rows * cols, euclidian_observation_ndim)}."
                              f"But the computed point cloud had shape {points.shape}. "
                              f"Please ensure input are correct. If this is an error, please submit an issue.")
    return points
