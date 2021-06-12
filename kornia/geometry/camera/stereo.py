import torch
from kornia.geometry.conversions import convert_points_from_homogeneous


class StereoCameraException(Exception):
    def __init__(self, msg, *args, **kwargs):
        doc_help = "\n Please check documents here: ... for further information and examples."
        final_msg = msg + doc_help
        super().__init__(final_msg, *args, **kwargs)


class StereoCamera:
    def __init__(self, rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor):
        """

        Args:
            rectified_left_camera:
            rectified_right_camera:
        """
        self._check_stereo_camera(rectified_left_camera, rectified_right_camera)
        self.rectified_left_camera: torch.Tensor = rectified_left_camera
        self.rectified_right_camera: torch.Tensor = rectified_right_camera

        self.device: torch.device = self.rectified_left_camera.device
        self.dtype: torch.dtype = self.rectified_left_camera.dtype

    @staticmethod
    def _check_stereo_camera(rectified_left_camera: torch.Tensor, rectified_right_camera: torch.Tensor):
        """
        """
        # Ensure correct shapes
        if rectified_left_camera.ndim != 3:
            raise StereoCameraException("")

        if rectified_right_camera.ndim != 3:
            raise StereoCameraException("")

        if rectified_left_camera.shape[:1] == (3, 4):
            raise StereoCameraException("")

        if rectified_right_camera.shape[:1] == (3, 4):
            raise StereoCameraException("")

        # Ensure same devices for cameras.
        if rectified_left_camera.device != rectified_right_camera.device:
            raise StereoCameraException("")

        # Ensure same dtypes for cameras.
        if rectified_left_camera.dtype != rectified_right_camera.dtype:
            raise StereoCameraException("")

        # Ensure all intrinsics parameters (fx, fy, cx, cy) are the same in both cameras.
        if not torch.all(torch.eq(rectified_left_camera[..., :, :3], rectified_right_camera[..., :, :3])):
            raise StereoCameraException("Not equal intrinsics.")

        # Ensure that tx * fx is negative and exists.
        tx_fx = rectified_right_camera[..., 0, 3]
        if torch.all(torch.gt(tx_fx, 0)):
            raise StereoCameraException("")

        # Ensure we don't have a vertical stereo setup.
        ty_fy = rectified_right_camera[..., 1, 3]
        if not torch.all(torch.eq(ty_fy, 0)):
            raise StereoCameraException("")

    def _check_disparity_tensor(self, disparity_tensor: torch.Tensor):
        if disparity_tensor.shape[0] != self.batch_size:
            raise StereoCameraException("")

        if disparity_tensor.ndim != 3:
            raise StereoCameraException("")

        if disparity_tensor.dtype != torch.float32:
            raise StereoCameraException("")

    @property
    def batch_size(self) -> int:
        r"""Returns the batch size of the storage.

        Returns:
            int: scalar with the batch size
        """
        return self.rectified_left_camera.shape[0]

    @property
    def fx(self) -> torch.Tensor:
        r"""Returns the focal lenght in the x-direction.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        r"""Returns the focal length in the y-direction.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 1]

    @property
    def cx(self) -> torch.Tensor:
        r"""Returns the x-coordinate of the principal point.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        r"""Returns the y-coordinate of the principal point.

        Returns:
            torch.Tensor: tensor of shape :math:`(B)`
        """
        return self.rectified_left_camera[..., 1, 2]

    @property
    def tx(self) -> torch.Tensor:
        return self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> torch.Tensor:
        """
            [ 1 0   0      -Cx      ]
        Q = [ 0 1   0      -Cy      ]
            [ 0 0   0       Fx      ]
            [ 0 0 -1/Tx (Cx-Cx')/Tx ]
        https://github.com/opencv/opencv/blob/438e2dc22802e801b65683f7df46eca62b625474/modules/calib3d/src/calibration.cpp#L2698
        """
        Q = torch.zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        Q[:, 0, 0] = 1
        Q[:, 1, 1] = 1
        Q[:, 0, 3] = -self.cx
        Q[:, 1, 3] = -self.cy
        Q[:, 2, 3] = self.fx
        Q[:, 3, 2] = -1 / self.tx
        Q[:, 3, 3] = 0  # TODO: Correct for different principal points
        return Q

    def reproject_disparity_to_3D(self, disparity_tensor):
        """

        Args:
            disparity_tensor:

        Returns:

        """
        self._check_disparity_tensor(disparity_tensor)
        return reproject_disparity_to_3D(disparity_tensor, self.Q)


def reproject_disparity_to_3D(disparity_tensor, Q_matrix):
    """

    Args:
        disparity_tensor:
        Q_matrix:

    Returns:

    """
    batch_size, rows, cols = disparity_tensor.shape
    if not Q_matrix.shape == (batch_size, 4, 4):
        raise StereoCameraException("")
    dtype = disparity_tensor.dtype
    device = disparity_tensor.device
    x, y = torch.meshgrid(
        torch.arange(rows, dtype=dtype, device=device),
        torch.arange(cols, dtype=dtype, device=device))
    x = x.expand(batch_size, -1, -1)
    y = y.expand(batch_size, -1, -1)
    z = torch.ones((batch_size, rows, cols), dtype=dtype, device=device)
    xydz = torch.stack((x, y, disparity_tensor, z), -1).permute(0, 3, 1, 2).reshape(batch_size, 4, -1)
    hom_points = torch.bmm(Q_matrix, xydz)
    return convert_points_from_homogeneous(hom_points.permute(0, 2, 1)).reshape(batch_size, rows, cols, 3)
