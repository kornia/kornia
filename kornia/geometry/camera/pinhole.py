from typing import Iterable, List, Union

import torch

from kornia.core import Device, Tensor, eye, stack, zeros
from kornia.core.check import KORNIA_CHECK_SAME_DEVICE
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.linalg import inverse_transformation, transform_points
from kornia.utils.helpers import _torch_inverse_cast


class PinholeCamera:
    r"""Class that represents a Pinhole Camera model.

    Args:
        intrinsics: tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 camera calibration matrix.
        extrinsics: tensor with shape :math:`(B, 4, 4)`
          containing the full 4x4 rotation-translation matrix.
        height: tensor with shape :math:`(B)` containing the image height.
        width: tensor with shape :math:`(B)` containing the image width.

    .. note::
        We assume that the class attributes are in batch form in order to take
        advantage of PyTorch parallelism to boost computing performance.
    """

    def __init__(self, intrinsics: Tensor, extrinsics: Tensor, height: Tensor, width: Tensor) -> None:
        # verify batch size and shapes
        self._check_valid([intrinsics, extrinsics, height, width])
        self._check_valid_params(intrinsics, "intrinsics")
        self._check_valid_params(extrinsics, "extrinsics")
        self._check_valid_shape(height, "height")
        self._check_valid_shape(width, "width")
        self._check_consistent_device([intrinsics, extrinsics, height, width])
        # set class attributes
        self.height: Tensor = height
        self.width: Tensor = width
        self._intrinsics: Tensor = intrinsics
        self._extrinsics: Tensor = extrinsics

    @staticmethod
    def _check_valid(data_iter: Iterable[Tensor]) -> bool:
        if not all(data.shape[0] for data in data_iter):
            raise ValueError("Arguments shapes must match")
        return True

    @staticmethod
    def _check_valid_params(data: Tensor, data_name: str) -> bool:
        if len(data.shape) not in (3, 4) and data.shape[-2:] != (4, 4):  # Shouldn't this be an OR logic than AND?
            raise ValueError(
                f"Argument {data_name} shape must be in the following shape Bx4x4 or BxNx4x4. Got {data.shape}"
            )
        return True

    @staticmethod
    def _check_valid_shape(data: Tensor, data_name: str) -> bool:
        if not len(data.shape) == 1:
            raise ValueError(f"Argument {data_name} shape must be in the following shape B. Got {data.shape}")
        return True

    @staticmethod
    def _check_consistent_device(data_iter: List[Tensor]) -> None:
        first = data_iter[0]
        for data in data_iter:
            KORNIA_CHECK_SAME_DEVICE(data, first)

    def device(self) -> torch.device:
        r"""Returns the device for camera buffers.

        Returns:
            Device type
        """
        return self._intrinsics.device

    @property
    def intrinsics(self) -> Tensor:
        r"""The full 4x4 intrinsics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        if not self._check_valid_params(self._intrinsics, "intrinsics"):
            raise AssertionError
        return self._intrinsics

    @property
    def extrinsics(self) -> Tensor:
        r"""The full 4x4 extrinsics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        if not self._check_valid_params(self._extrinsics, "extrinsics"):
            raise AssertionError
        return self._extrinsics

    @property
    def batch_size(self) -> int:
        r"""Return the batch size of the storage.

        Returns:
            scalar with the batch size.
        """
        return self.intrinsics.shape[0]

    @property
    def fx(self) -> Tensor:
        r"""Return the focal length in the x-direction.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 0, 0]

    @property
    def fy(self) -> Tensor:
        r"""Return the focal length in the y-direction.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 1, 1]

    @property
    def cx(self) -> Tensor:
        r"""Return the x-coordinate of the principal point.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 0, 2]

    @property
    def cy(self) -> Tensor:
        r"""Return the y-coordinate of the principal point.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.intrinsics[..., 1, 2]

    @property
    def tx(self) -> Tensor:
        r"""Return the x-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 0, -1]

    @tx.setter
    def tx(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the x-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 0, -1] = value
        return self

    @property
    def ty(self) -> Tensor:
        r"""Return the y-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 1, -1]

    @ty.setter
    def ty(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the y-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 1, -1] = value
        return self

    @property
    def tz(self) -> Tensor:
        r"""Returns the z-coordinate of the translation vector.

        Returns:
            tensor of shape :math:`(B)`.
        """
        return self.extrinsics[..., 2, -1]

    @tz.setter
    def tz(self, value: Union[Tensor, float]) -> "PinholeCamera":
        r"""Set the y-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 2, -1] = value
        return self

    @property
    def rt_matrix(self) -> Tensor:
        r"""Return the 3x4 rotation-translation matrix.

        Returns:
            tensor of shape :math:`(B, 3, 4)`.
        """
        return self.extrinsics[..., :3, :4]

    @property
    def camera_matrix(self) -> Tensor:
        r"""Return the 3x3 camera matrix containing the intrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 3)`.
        """
        return self.intrinsics[..., :3, :3]

    @property
    def rotation_matrix(self) -> Tensor:
        r"""Return the 3x3 rotation matrix from the extrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 3)`.
        """
        return self.extrinsics[..., :3, :3]

    @property
    def translation_vector(self) -> Tensor:
        r"""Return the translation vector from the extrinsics.

        Returns:
            tensor of shape :math:`(B, 3, 1)`.
        """
        return self.extrinsics[..., :3, -1:]

    def clone(self) -> "PinholeCamera":
        r"""Return a deep copy of the current object instance."""
        height: Tensor = self.height.clone()
        width: Tensor = self.width.clone()
        intrinsics: Tensor = self.intrinsics.clone()
        extrinsics: Tensor = self.extrinsics.clone()
        return PinholeCamera(intrinsics, extrinsics, height, width)

    def intrinsics_inverse(self) -> Tensor:
        r"""Return the inverse of the 4x4 instrisics matrix.

        Returns:
            tensor of shape :math:`(B, 4, 4)`.
        """
        return self.intrinsics.inverse()

    def scale(self, scale_factor: Tensor) -> "PinholeCamera":
        r"""Scale the pinhole model.

        Args:
            scale_factor: a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            the camera model with scaled parameters.
        """
        # scale the intrinsic parameters
        intrinsics: Tensor = self.intrinsics.clone()
        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        height: Tensor = scale_factor * self.height.clone()
        width: Tensor = scale_factor * self.width.clone()
        return PinholeCamera(intrinsics, self.extrinsics, height, width)

    def scale_(self, scale_factor: Union[float, Tensor]) -> "PinholeCamera":
        r"""Scale the pinhole model in-place.

        Args:
            scale_factor: a tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              :math:`(B)` or :math:`(1)`.

        Returns:
            the camera model with scaled parameters.
        """
        # scale the intrinsic parameters
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        self.height *= scale_factor
        self.width *= scale_factor
        return self

    def project(self, point_3d: Tensor) -> Tensor:
        r"""Project a 3d point in world coordinates onto the 2d camera plane.

        Args:
            point3d: tensor containing the 3d points to be projected
                to the camera plane. The shape of the tensor can be :math:`(*, 3)`.

        Returns:
            tensor of (u, v) cam coordinates with shape :math:`(*, 2)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> X = torch.rand(1, 3)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.project(X)
            tensor([[5.6088, 8.6827]])
        """
        P = self.intrinsics @ self.extrinsics
        return convert_points_from_homogeneous(transform_points(P, point_3d))

    def unproject(self, point_2d: Tensor, depth: Tensor) -> Tensor:
        r"""Unproject a 2d point in 3d.

        Transform coordinates in the pixel frame to the world frame.

        Args:
            point2d: tensor containing the 2d to be projected to
                world coordinates. The shape of the tensor can be :math:`(*, 2)`.
            depth: tensor containing the depth value of each 2d
                points. The tensor shape must be equal to point2d :math:`(*, 1)`.
            normalize: whether to normalize the pointcloud. This
                must be set to `True` when the depth is represented as the Euclidean
                ray length from the camera position.

        Returns:
            tensor of (x, y, z) world coordinates with shape :math:`(*, 3)`.

        Example:
            >>> _ = torch.manual_seed(0)
            >>> x = torch.rand(1, 2)
            >>> depth = torch.ones(1, 1)
            >>> K = torch.eye(4)[None]
            >>> E = torch.eye(4)[None]
            >>> h = torch.ones(1)
            >>> w = torch.ones(1)
            >>> pinhole = kornia.geometry.camera.PinholeCamera(K, E, h, w)
            >>> pinhole.unproject(x, depth)
            tensor([[0.4963, 0.7682, 1.0000]])
        """
        P = self.intrinsics @ self.extrinsics
        P_inv = _torch_inverse_cast(P)
        return transform_points(P_inv, convert_points_to_homogeneous(point_2d) * depth)

    # NOTE: just for test. Decide if we keep it.
    @classmethod
    def from_parameters(
        self,
        fx: Tensor,
        fy: Tensor,
        cx: Tensor,
        cy: Tensor,
        height: int,
        width: int,
        tx: Tensor,
        ty: Tensor,
        tz: Tensor,
        batch_size: int,
        device: Device,
        dtype: torch.dtype,
    ) -> "PinholeCamera":
        # create the camera matrix
        intrinsics = zeros(batch_size, 4, 4, device=device, dtype=dtype)
        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0
        # create the pose matrix
        extrinsics = eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        extrinsics[..., 0, -1] += tx
        extrinsics[..., 1, -1] += ty
        extrinsics[..., 2, -1] += tz
        # create image hegith and width
        height_tmp = zeros(batch_size, device=device, dtype=dtype)
        height_tmp[..., 0] += height
        width_tmp = zeros(batch_size, device=device, dtype=dtype)
        width_tmp[..., 0] += width
        return self(intrinsics, extrinsics, height_tmp, width_tmp)


class PinholeCamerasList(PinholeCamera):
    r"""Class that represents a list of pinhole cameras.

    The class inherits from :class:`~kornia.PinholeCamera` meaning that
    it will keep the same class properties but with an extra dimension.

    .. note::
        The underlying data tensor will be stacked in the first dimension.
        That's it, given a list of two camera instances, the intrinsics tensor
        will have a shape :math:`(B, N, 4, 4)` where :math:`B` is the batch
        size and :math:`N` is the numbers of cameras (in this case two).

    Args:
        pinholes_list: a python tuple or list containing a set of `PinholeCamera` instances.
    """

    def __init__(self, pinholes_list: Iterable[PinholeCamera]) -> None:
        self._initialize_parameters(pinholes_list)

    def _initialize_parameters(self, pinholes: Iterable[PinholeCamera]) -> "PinholeCamerasList":
        r"""Initialise the class attributes given a cameras list."""
        if not isinstance(pinholes, (list, tuple)):
            raise TypeError(f"pinhole must of type list or tuple. Got {type(pinholes)}")
        height, width = [], []
        intrinsics, extrinsics = [], []
        for pinhole in pinholes:
            if not isinstance(pinhole, PinholeCamera):
                raise TypeError(f"Argument pinhole must be from type PinholeCamera. Got {type(pinhole)}")
            height.append(pinhole.height)
            width.append(pinhole.width)
            intrinsics.append(pinhole.intrinsics)
            extrinsics.append(pinhole.extrinsics)
        # concatenate and set members. We will assume BxNx4x4
        self.height: Tensor = stack(height, dim=1)
        self.width: Tensor = stack(width, dim=1)
        self._intrinsics: Tensor = stack(intrinsics, dim=1)
        self._extrinsics: Tensor = stack(extrinsics, dim=1)
        return self

    @property
    def num_cameras(self) -> int:
        r"""Return the number of pinholes cameras per batch."""
        num_cameras: int = -1
        if self.intrinsics is not None:
            num_cameras = int(self.intrinsics.shape[1])
        return num_cameras

    def get_pinhole(self, idx: int) -> PinholeCamera:
        r"""Return a PinholeCamera object with parameters such as Bx4x4."""
        height: Tensor = self.height[..., idx]
        width: Tensor = self.width[..., idx]
        intrinsics: Tensor = self.intrinsics[:, idx]
        extrinsics: Tensor = self.extrinsics[:, idx]
        return PinholeCamera(intrinsics, extrinsics, height, width)


def pinhole_matrix(pinholes: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Function that returns the pinhole matrix from a pinhole model.

    .. note::
        This method is going to be deprecated in version 0.2 in favour of
        :attr:`kornia.PinholeCamera.camera_matrix`.

    Args:
        pinholes: tensor of pinhole models.

    Returns:
        tensor of pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> rng = torch.manual_seed(0)
        >>> pinhole = torch.rand(1, 12)    # Nx12
        >>> pinhole_matrix(pinhole)  # Nx4x4
        tensor([[[4.9626e-01, 1.0000e-06, 8.8477e-02, 1.0000e-06],
                 [1.0000e-06, 7.6822e-01, 1.3203e-01, 1.0000e-06],
                 [1.0000e-06, 1.0000e-06, 1.0000e+00, 1.0000e-06],
                 [1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e+00]]])
    """
    # warnings.warn("pinhole_matrix will be deprecated in version 0.2, "
    #              "use PinholeCamera.camera_matrix instead",
    #              PendingDeprecationWarning)
    if not (len(pinholes.shape) == 2 and pinholes.shape[1] == 12):
        raise AssertionError(pinholes.shape)
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinholes[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = eye(4, device=pinholes.device, dtype=pinholes.dtype) + eps
    k = k.view(1, 4, 4).repeat(pinholes.shape[0], 1, 1)  # Nx4x4
    # fill output with pinhole values
    k[..., 0, 0:1] = fx
    k[..., 0, 2:3] = cx
    k[..., 1, 1:2] = fy
    k[..., 1, 2:3] = cy
    return k


def inverse_pinhole_matrix(pinhole: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Return the inverted pinhole matrix from a pinhole model.

    .. note::
        This method is going to be deprecated in version 0.2 in favour of
        :attr:`kornia.PinholeCamera.intrinsics_inverse()`.

    Args:
        pinholes: tensor with pinhole models.

    Returns:
        tensor of inverted pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> rng = torch.manual_seed(0)
        >>> pinhole = torch.rand(1, 12)  # Nx12
        >>> inverse_pinhole_matrix(pinhole)  # Nx4x4
        tensor([[[ 2.0151,  0.0000, -0.1783,  0.0000],
                 [ 0.0000,  1.3017, -0.1719,  0.0000],
                 [ 0.0000,  0.0000,  1.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    """
    # warnings.warn("inverse_pinhole_matrix will be deprecated in version 0.2, "
    #              "use PinholeCamera.intrinsics_inverse() instead",
    #              PendingDeprecationWarning)
    if not (len(pinhole.shape) == 2 and pinhole.shape[1] == 12):
        raise AssertionError(pinhole.shape)
    # unpack pinhole values
    fx, fy, cx, cy = torch.chunk(pinhole[..., :4], 4, dim=1)  # Nx1
    # create output container
    k = eye(4, device=pinhole.device, dtype=pinhole.dtype)
    k = k.view(1, 4, 4).repeat(pinhole.shape[0], 1, 1)  # Nx4x4
    # fill output with inverse values
    k[..., 0, 0:1] = 1.0 / (fx + eps)
    k[..., 1, 1:2] = 1.0 / (fy + eps)
    k[..., 0, 2:3] = -1.0 * cx / (fx + eps)
    k[..., 1, 2:3] = -1.0 * cy / (fy + eps)
    return k


def scale_pinhole(pinholes: Tensor, scale: Tensor) -> Tensor:
    r"""Scale the pinhole matrix for each pinhole model.

    .. note::
        This method is going to be deprecated in version 0.2 in favour of
        :attr:`kornia.PinholeCamera.scale()`.

    Args:
        pinholes: tensor with the pinhole model.
        scale: tensor of scales.

    Returns:
        tensor of scaled pinholes.

    Shape:
        - Input: :math:`(N, 12)` and :math:`(N, 1)`
        - Output: :math:`(N, 12)`

    Example:
        >>> rng = torch.manual_seed(0)
        >>> pinhole_i = torch.rand(1, 12)  # Nx12
        >>> scales = 2.0 * torch.ones(1)   # N
        >>> scale_pinhole(pinhole_i, scales)  # Nx12
        tensor([[0.9925, 1.5364, 0.1770, 0.2641, 0.6148, 1.2682, 0.4901, 0.8964, 0.4556,
                 0.6323, 0.3489, 0.4017]])
    """
    # warnings.warn("scale_pinhole will be deprecated in version 0.2, "
    #              "use PinholeCamera.scale() instead",
    #              PendingDeprecationWarning)
    if not (len(pinholes.shape) == 2 and pinholes.shape[1] == 12):
        raise AssertionError(pinholes.shape)
    if len(scale.shape) != 1:
        raise AssertionError(scale.shape)
    pinholes_scaled = pinholes.clone()
    pinholes_scaled[..., :6] = pinholes[..., :6] * scale.unsqueeze(-1)
    return pinholes_scaled


def get_optical_pose_base(pinholes: Tensor) -> Tensor:
    """Compute extrinsic transformation matrices for pinholes.

    Args:
        pinholes: tensor of form [fx fy cx cy h w rx ry rz tx ty tz]
                           of size (N, 12).

    Returns:
        tensor of extrinsic transformation matrices of size (N, 4, 4).
    """
    if not (len(pinholes.shape) == 2 and pinholes.shape[1] == 12):
        raise AssertionError(pinholes.shape)
    # TODO: where is rtvec_to_pose?
    raise NotImplementedError
    # TODO: We have rtvec_to_pose in torchgeometry
    # https://github.com/whh14/torchgeometry/blob/master/torchgeometry/conversions.py#L240
    # But it relies on axis_angle_to_rotation_matrix
    # And since then, it was changed from returning Nx4x4 matrix to Nx3x3
    # return rtvec_to_pose(optical_pose_parent)   type: ignore


def homography_i_H_ref(pinhole_i: Tensor, pinhole_ref: Tensor) -> Tensor:
    r"""Homography from reference to ith pinhole.

    .. note::
        The pinhole model is represented in a single vector as follows:

        .. math::
            pinhole = (f_x, f_y, c_x, c_y, height, width,
            r_x, r_y, r_z, t_x, t_y, t_z)

        where:
            :math:`(r_x, r_y, r_z)` is the rotation vector in angle-axis
            convention.

            :math:`(t_x, t_y, t_z)` is the translation vector.

    .. math::

        H_{ref}^{i} = K_{i} * T_{ref}^{i} * K_{ref}^{-1}

    Args:
        pinhole_i: tensor with pinhole model for ith frame.
        pinhole_ref: tensor with pinhole model for reference frame.

    Returns:
        tensors that convert depth points (u, v, d) from pinhole_ref to pinhole_i.

    Shape:
        - Input: :math:`(N, 12)` and :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        pinhole_i = torch.rand(1, 12)    # Nx12
        pinhole_ref = torch.rand(1, 12)  # Nx12
        homography_i_H_ref(pinhole_i, pinhole_ref)  # Nx4x4
    """
    # TODO: Add doctest once having `rtvec_to_pose`.
    if not (len(pinhole_i.shape) == 2 and pinhole_i.shape[1] == 12):
        raise AssertionError(pinhole_i.shape)
    if pinhole_i.shape != pinhole_ref.shape:
        raise AssertionError(pinhole_ref.shape)
    i_pose_base = get_optical_pose_base(pinhole_i)
    ref_pose_base = get_optical_pose_base(pinhole_ref)
    i_pose_ref = torch.matmul(i_pose_base, inverse_transformation(ref_pose_base))
    return torch.matmul(pinhole_matrix(pinhole_i), torch.matmul(i_pose_ref, inverse_pinhole_matrix(pinhole_ref)))


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L26


def pixel2cam(depth: Tensor, intrinsics_inv: Tensor, pixel_coords: Tensor) -> Tensor:
    r"""Transform coordinates in the pixel frame to the camera frame.

    Args:
        depth: the source depth maps. Shape must be Bx1xHxW.
        intrinsics_inv: the inverse intrinsics camera matrix. Shape must be Bx4x4.
        pixel_coords: the grid with (u, v, 1) pixel coordinates. Shape must be BxHxWx3.

    Returns:
        tensor of shape BxHxWx3 with (x, y, z) cam coordinates.
    """
    if not len(depth.shape) == 4 and depth.shape[1] == 1:
        raise ValueError(f"Input depth has to be in the shape of Bx1xHxW. Got {depth.shape}")
    if not len(intrinsics_inv.shape) == 3:
        raise ValueError(f"Input intrinsics_inv has to be in the shape of Bx4x4. Got {intrinsics_inv.shape}")
    if not len(pixel_coords.shape) == 4 and pixel_coords.shape[3] == 3:
        raise ValueError(f"Input pixel_coords has to be in the shape of BxHxWx3. Got {intrinsics_inv.shape}")
    cam_coords: Tensor = transform_points(intrinsics_inv[:, None], pixel_coords)
    return cam_coords * depth.permute(0, 2, 3, 1)


# based on
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43


def cam2pixel(cam_coords_src: Tensor, dst_proj_src: Tensor, eps: float = 1e-12) -> Tensor:
    r"""Transform coordinates in the camera frame to the pixel frame.

    Args:
        cam_coords: (x, y, z) coordinates defined in the first camera coordinates system. Shape must be BxHxWx3.
        dst_proj_src: the projection matrix between the
          reference and the non reference camera frame. Shape must be Bx4x4.
        eps: small value to avoid division by zero error.

    Returns:
        tensor of shape BxHxWx2 with (u, v) pixel coordinates.
    """
    if not len(cam_coords_src.shape) == 4 and cam_coords_src.shape[3] == 3:
        raise ValueError(f"Input cam_coords_src has to be in the shape of BxHxWx3. Got {cam_coords_src.shape}")
    if not len(dst_proj_src.shape) == 3 and dst_proj_src.shape[-2:] == (4, 4):
        raise ValueError(f"Input dst_proj_src has to be in the shape of Bx4x4. Got {dst_proj_src.shape}")
    # apply projection matrix to points
    point_coords: Tensor = transform_points(dst_proj_src[:, None], cam_coords_src)
    x_coord: Tensor = point_coords[..., 0]
    y_coord: Tensor = point_coords[..., 1]
    z_coord: Tensor = point_coords[..., 2]

    # compute pixel coordinates
    u_coord: Tensor = x_coord / (z_coord + eps)
    v_coord: Tensor = y_coord / (z_coord + eps)

    # stack and return the coordinates, that's the actual flow
    pixel_coords_dst: Tensor = stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst  # BxHxWx2


# layer api
'''class PinholeMatrix(nn.Module):
    r"""Create an object that returns the pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor of pinhole models.

    Returns:
        Tensor: tensor of pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)          # Nx12
        >>> transform = PinholeMatrix()
        >>> pinhole_matrix = transform(pinhole)  # Nx4x4
    """

    def __init__(self):
        super(PinholeMatrix, self).__init__()

    def forward(self, input):
        return pinhole_matrix(input)


class InversePinholeMatrix(nn.Module):
    r"""Return and object that inverts a pinhole matrix from a pinhole model

    Args:
        pinholes (Tensor): tensor with pinhole models.

    Returns:
        Tensor: tensor of inverted pinhole matrices.

    Shape:
        - Input: :math:`(N, 12)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pinhole = torch.rand(1, 12)              # Nx12
        >>> transform = kornia.InversePinholeMatrix()
        >>> pinhole_matrix_inv = transform(pinhole)  # Nx4x4
    """

    def __init__(self):
        super(InversePinholeMatrix, self).__init__()

    def forward(self, input):
        return inverse_pinhole_matrix(input)'''
