import torch


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

    @staticmethod
    def _check_depth_tensor(depth_tensor: torch.Tensor):
        pass

    @staticmethod
    def _check_disparity_tensor(disparity_tensor: torch.Tensor):
        pass

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
        return -self.rectified_right_camera[..., 0, 3] / self.fx

    @property
    def Q(self) -> torch.Tensor:
        """
        """
        Q = torch.zeros((self.batch_size, 4, 4), device=self.device, dtype=self.dtype)
        Q[:, 0, 0] = self.fy * self.tx
        Q[:, 0, 3] = -self.fy * self.cx * self.tx
        Q[:, 1, 1] = self.fx * self.tx
        Q[:, 1, 3] = -self.fx * self.cy * self.tx
        Q[:, 2, 3] = self.fx * self.fy * self.tx
        Q[:, 3, 2] = -self.fy
        Q[:, 3, 3] = self.fy
        return Q