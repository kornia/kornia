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

"""Pinhole cameras with extrinsics: project world points to pixels and back, and the dense helpers that use them."""

import warnings
from typing import Iterable, List, Union

import torch

from kornia.core.check import KORNIA_CHECK_SAME_DEVICE
from kornia.core.utils import _torch_inverse_cast, is_compiling
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.linalg import transform_points


def _embed_intrinsics(camera_matrix: torch.Tensor) -> torch.Tensor:
    """Embed a (..., 3, 3) camera matrix in homogeneous 4x4 form."""
    intrinsics = camera_matrix.new_zeros(*camera_matrix.shape[:-2], 4, 4)
    intrinsics[..., :3, :3] = camera_matrix
    intrinsics[..., 3, 3] = 1.0
    return intrinsics


class PinholeCamera:
    r"""Model a batch of pinhole cameras with extrinsics: world points to pixels and back.

    ``cam = PinholeCamera(intrinsics, extrinsics, height, width)`` stores ``B`` cameras: ``(B, 4, 4)`` intrinsics,
    ``(B, 4, 4)`` world-to-camera extrinsics and the ``(B,)`` image height and width. ``uv = cam.project(points)``
    takes world points ``(B, N, 3)``, camera ``b`` projecting ``points[b]``, and returns their pixels
    ``(B, N, 2)``; ``points = cam.unproject(uv, depth)`` takes pixels ``(B, N, 2)`` and their camera-frame depths
    ``(B, N, 1)`` and returns world points ``(B, N, 3)``. For camera-frame points and a ``3x3`` ``K`` without
    extrinsics, use :func:`~kornia.geometry.camera.perspective.project_points` and
    :func:`~kornia.geometry.camera.perspective.unproject_points` instead.

    Convention:
        - ``intrinsics`` is the ``(B, 4, 4)`` calibration matrix whose top-left :math:`3 \times 3` block is
          ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, and ``extrinsics`` the ``(B, 4, 4)`` **world-to-camera**
          transform ``[R | t]`` (OpenCV / COLMAP semantics): :meth:`project` computes ``K (R X + t)`` of
          **world** points, and :meth:`unproject` returns **world** points. The functional API takes a ``3x3``
          ``K`` and no extrinsics, so it works in the **camera** frame.
        - with ``B > 1`` cameras the points of :meth:`project`, ``(B, N, 3)`` or ``(B, H, W, 3)``, and the pixels
          of :meth:`unproject`, ``(B, N, 2)`` or ``(B, H, W, 2)``, need a leading axis of size ``B`` and at least
          three dimensions: ``(N, 3)`` and ``(1, N, 3)`` points, or ``(N, 2)`` and ``(1, N, 2)`` pixels, raise
          :class:`ValueError`. A single camera (``B = 1``) takes any ``(*, N, 3)`` points or ``(*, N, 2)`` pixels
          with at least two dimensions.
        - pixel coordinates are ``(u, v)`` = (column, row) with **integer pixel centres**: pixel ``(0, 0)`` is
          centred at ``(0, 0)``, as :func:`~kornia.geometry.grid.create_meshgrid` enumerates, so a centred image
          has ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``. COLMAP's half-pixel convention reports the same
          principal point half a pixel larger. ``kornia.geometry.camera``, ``kornia.geometry.depth`` and
          ``kornia.geometry.calibration`` all use this convention; see :doc:`/get-started/camera-conventions`.
        - ``depth`` is the camera-frame ``z``, except where a ``normalize`` / ``normalize_points`` flag reads it
          as the Euclidean ray length.
        - the class owns copies of the tensors it is constructed from: :meth:`scale_` and the ``tx`` / ``ty`` /
          ``tz`` setters update only the camera, and :meth:`scale` and :meth:`clone` return independent cameras.

    .. warning::
        :meth:`scale` and :meth:`scale_` rescale the principal point as ``cx' = s * cx`` (the half-pixel rule),
        which disagrees with the integer pixel centres above:
        `#4263 <https://github.com/kornia/kornia/issues/4263>`_. ``(B, N, 4, 4)`` storage is
        deprecated and will be rejected in a future release:
        `#4266 <https://github.com/kornia/kornia/issues/4266>`_. The ``intrinsics`` form is not validated, but
        :meth:`project`, :meth:`unproject`, :meth:`intrinsics_inverse` and
        :class:`~kornia.geometry.depth.DepthWarper` read only the top-left :math:`3 \times 3` block, so a
        zero-padded ``K`` behaves like its homogeneous embedding.

    Args:
        intrinsics: torch.Tensor with shape ``(B, 4, 4)``
          containing the full 4x4 camera calibration matrix. Pass ``(B, 4, 4)``:
          ``(B, N, 4, 4)`` is deprecated, and :meth:`project` and :meth:`unproject` never supported it.
          ``PinholeCamerasList`` still stores that layout internally.
        extrinsics: torch.Tensor with shape ``(B, 4, 4)``
          containing the full 4x4 rotation-translation matrix, checked by the same predicate.
        height: torch.Tensor with shape ``(B)`` containing the image height.
        width: torch.Tensor with shape ``(B)`` containing the image width.

    .. note::
        We assume that the class attributes are in batch form in order to take
        advantage of PyTorch parallelism to boost computing performance.

    See Also:
        - :func:`~kornia.geometry.camera.perspective.project_points`: camera-frame points and a ``(*, 3, 3)`` ``K``,
          with no extrinsics.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: pixels at given depths to camera-frame
          points with a ``(*, 3, 3)`` ``K``.
        - :func:`pixel2cam`, :func:`cam2pixel`: the two halves of a dense reprojection, a depth map to camera-frame
          points and those to another camera's pixels, with this class's ``(B, 4, 4)`` matrices.
        - :class:`~kornia.geometry.depth.DepthWarper`: warps an image between two of these cameras by depth.
        - :class:`~kornia.geometry.camera.stereo.StereoCamera`: a rectified stereo pair, from its two ``(B, 3, 4)``
          projection matrices.
        - :class:`~kornia.sensors.camera.CameraModel`: the experimental camera-model API, with lens distortion and
          flat parameter vectors but no extrinsics.

    """

    def __init__(
        self, intrinsics: torch.Tensor, extrinsics: torch.Tensor, height: torch.Tensor, width: torch.Tensor
    ) -> None:
        # verify batch size and shapes
        self._check_valid([intrinsics, extrinsics, height, width])
        self._check_valid_params(intrinsics, "intrinsics")
        self._check_valid_params(extrinsics, "extrinsics")
        is_rank4 = intrinsics.dim() == 4 or extrinsics.dim() == 4
        if is_rank4 and not torch.jit.is_scripting() and not torch.jit.is_tracing() and not is_compiling():
            warnings.warn(
                "Passing (B, N, 4, 4) `intrinsics` or `extrinsics` to `PinholeCamera` is deprecated since kornia "
                "0.9.1 and will raise a ValueError in a future release: `project` and `unproject` support only "
                "(B, 4, 4). Flatten the cameras into the batch dimension, for example "
                "`intrinsics.reshape(-1, 4, 4)` and `extrinsics.reshape(-1, 4, 4)` (expanding a (B, 4, 4) "
                "matrix across N first), and repeat `height` and `width` to match.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._check_valid_shape(height, "height")
        self._check_valid_shape(width, "width")
        self._check_consistent_device([intrinsics, extrinsics, height, width])
        # set class attributes
        self.height: torch.Tensor = height.clone()
        self.width: torch.Tensor = width.clone()
        self._intrinsics: torch.Tensor = intrinsics.clone()
        self._extrinsics: torch.Tensor = extrinsics.clone()

    @staticmethod
    def _check_valid(data_iter: Iterable[torch.Tensor]) -> bool:
        batch_sizes = [data.shape[0] for data in data_iter]
        if not all(batch_size == batch_sizes[0] for batch_size in batch_sizes):
            raise ValueError("Arguments shapes must match")
        return True

    @staticmethod
    def _check_valid_params(data: torch.Tensor, data_name: str) -> bool:
        if len(data.shape) not in (3, 4) or data.shape[-2:] != (4, 4):
            raise ValueError(f"Argument {data_name} shape must be Bx4x4 or BxNx4x4. Got {data.shape}")
        return True

    @staticmethod
    def _check_valid_shape(data: torch.Tensor, data_name: str) -> bool:
        if not len(data.shape) == 1:
            raise ValueError(f"Argument {data_name} shape must be in the following shape B. Got {data.shape}")
        return True

    @staticmethod
    def _check_consistent_device(data_iter: List[torch.Tensor]) -> None:
        first = data_iter[0]
        for data in data_iter:
            KORNIA_CHECK_SAME_DEVICE(data, first)

    def device(self) -> torch.device:
        r"""Return the device for camera buffers.

        Returns:
            the ``torch.device`` of the stored intrinsics.

        """
        return self._intrinsics.device

    @property
    def intrinsics(self) -> torch.Tensor:
        r"""The full 4x4 intrinsics matrix.

        Returns:
            torch.Tensor of shape ``(B, 4, 4)``.

        """
        if not self._check_valid_params(self._intrinsics, "intrinsics"):
            raise AssertionError
        return self._intrinsics

    @property
    def extrinsics(self) -> torch.Tensor:
        r"""The full 4x4 extrinsics matrix.

        Returns:
            torch.Tensor of shape ``(B, 4, 4)``.

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
    def fx(self) -> torch.Tensor:
        r"""Return the focal length in the x-direction.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.intrinsics[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        r"""Return the focal length in the y-direction.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.intrinsics[..., 1, 1]

    @property
    def cx(self) -> torch.Tensor:
        r"""Return the x-coordinate of the principal point.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.intrinsics[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        r"""Return the y-coordinate of the principal point.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.intrinsics[..., 1, 2]

    @property
    def tx(self) -> torch.Tensor:
        r"""Return the x-coordinate of the translation vector.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.extrinsics[..., 0, -1]

    @tx.setter
    def tx(self, value: Union[torch.Tensor, float]) -> None:
        r"""Set the x-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 0, -1] = value

    @property
    def ty(self) -> torch.Tensor:
        r"""Return the y-coordinate of the translation vector.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.extrinsics[..., 1, -1]

    @ty.setter
    def ty(self, value: Union[torch.Tensor, float]) -> None:
        r"""Set the y-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 1, -1] = value

    @property
    def tz(self) -> torch.Tensor:
        r"""Returns the z-coordinate of the translation vector.

        Returns:
            torch.Tensor of shape ``(B)``.

        """
        return self.extrinsics[..., 2, -1]

    @tz.setter
    def tz(self, value: Union[torch.Tensor, float]) -> None:
        r"""Set the z-coordinate of the translation vector with the given value."""
        self.extrinsics[..., 2, -1] = value

    @property
    def rt_matrix(self) -> torch.Tensor:
        r"""Return the 3x4 rotation-translation matrix.

        Returns:
            torch.Tensor of shape ``(B, 3, 4)``.

        """
        return self.extrinsics[..., :3, :4]

    @property
    def camera_matrix(self) -> torch.Tensor:
        r"""Return the 3x3 camera matrix containing the intrinsics.

        Returns:
            torch.Tensor of shape ``(B, 3, 3)``.

        """
        return self.intrinsics[..., :3, :3]

    @property
    def rotation_matrix(self) -> torch.Tensor:
        r"""Return the 3x3 rotation matrix from the extrinsics.

        Returns:
            torch.Tensor of shape ``(B, 3, 3)``.

        """
        return self.extrinsics[..., :3, :3]

    @property
    def translation_vector(self) -> torch.Tensor:
        r"""Return the translation vector from the extrinsics.

        Returns:
            torch.Tensor of shape ``(B, 3, 1)``.

        """
        return self.extrinsics[..., :3, -1:]

    def clone(self) -> "PinholeCamera":
        r"""Return a deep copy of the current object instance.

        See the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        """
        return PinholeCamera(self.intrinsics, self.extrinsics, self.height, self.width)

    def intrinsics_inverse(self) -> torch.Tensor:
        r"""Return the homogeneous 4x4 embedding of the inverse camera matrix.

        Only the top-left 3x3 block of ``intrinsics`` is inverted. The remaining entries
        are ignored, so zero-padded intrinsics are supported. The result has a zero
        last row and column except for the homogeneous component at ``[..., 3, 3]``,
        which is one. See the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

        Returns:
            torch.Tensor of shape ``(B, 4, 4)``.

        """
        return _embed_intrinsics(_torch_inverse_cast(self.camera_matrix))

    def scale(self, scale_factor: torch.Tensor) -> "PinholeCamera":
        r"""Scale the pinhole model.

        Convention:
            - returns a **new** camera whose focal lengths, principal point and image size are multiplied by
              ``scale_factor``: ``fx' = s * fx`` and ``cx' = s * cx``.
            - ``height`` / ``width`` take the promoted dtype of ``scale_factor * height``.

        .. warning::
            The ``cx' = s * cx`` rule disagrees with kornia's integer pixel centres:
            `#4263 <https://github.com/kornia/kornia/issues/4263>`_.

        Args:
            scale_factor: a torch.Tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              ``(B)`` or ``(1)``.

        Returns:
            the camera model with scaled parameters.

        """
        # scale the intrinsic parameters
        intrinsics: torch.Tensor = self.intrinsics.clone()
        intrinsics[..., 0, 0] *= scale_factor
        intrinsics[..., 1, 1] *= scale_factor
        intrinsics[..., 0, 2] *= scale_factor
        intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        height: torch.Tensor = scale_factor * self.height.clone()
        width: torch.Tensor = scale_factor * self.width.clone()
        return PinholeCamera(intrinsics, self.extrinsics, height, width)

    def scale_(self, scale_factor: Union[float, torch.Tensor]) -> "PinholeCamera":
        r"""Scale the pinhole model in-place.

        Convention:
            - applies the rescaling of :meth:`scale` in place and returns ``self``; the tensors passed to the
              constructor are not modified.

        .. warning::
            The principal-point rule shared with :meth:`scale` is
            `#4263 <https://github.com/kornia/kornia/issues/4263>`_.

        Args:
            scale_factor: a torch.Tensor with the scale factor. It has
              to be broadcastable with class members. The expected shape is
              ``(B)`` or ``(1)``.

        Returns:
            the camera model with scaled parameters.

        """
        # scale the intrinsic parameters
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor
        # scale the image height/width
        self.height = self.height * scale_factor
        self.width = self.width * scale_factor
        return self

    def project(self, point_3d: torch.Tensor) -> torch.Tensor:
        r"""Project a 3d point in world coordinates onto the 2d camera plane.

        See the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

        Convention:
            - ``point_3d`` must be at least rank 2. An unbatched ``(3,)`` point raises :class:`ValueError`.
              An empty point set returns an empty result.
            - a point whose **camera-frame** ``z`` is 0 does not raise: ``K`` is applied first and the
              perspective divide is then skipped, so the result is the undivided ``K (R X + t)``.

        .. warning::
            This ``z = 0`` answer differs from :func:`~kornia.geometry.camera.perspective.project_points`, which
            applies ``K`` after the skipped divide: `#4267 <https://github.com/kornia/kornia/issues/4267>`_.

        Args:
            point_3d: torch.Tensor containing the 3d points to be projected
                to the camera plane. The shape of the torch.Tensor can be ``(*, 3)`` with at least two
                dimensions; with ``B > 1`` cameras, ``(B, *, N, 3)``, camera ``b`` projecting ``point_3d[b]``.

        Returns:
            torch.Tensor of (u, v) pixel coordinates with shape ``(*, 2)``.

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
        if len(point_3d.shape) < 2:
            raise ValueError(f"Input must be at least a 2D tensor. Got {point_3d.shape}")
        P = self._projection_matrix()
        return convert_points_from_homogeneous(transform_points(P, point_3d))

    def unproject(self, point_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        r"""Unproject a 2d point in 3d.

        Transform coordinates in the pixel frame to the world frame.

        See the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

        Args:
            point_2d: torch.Tensor containing the 2d to be projected to
                world coordinates. The shape of the torch.Tensor can be ``(*, 2)`` with at least two
                dimensions; with ``B > 1`` cameras, ``(B, *, N, 2)``, camera ``b`` unprojecting ``point_2d[b]``.
            depth: torch.Tensor containing the depth value of each 2d
                points. The torch.Tensor shape must be equal to point2d ``(*, 1)``.

        Returns:
            torch.Tensor of (x, y, z) world coordinates with shape ``(*, 3)``.

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
        P_inv = _torch_inverse_cast(self._projection_matrix())
        return transform_points(P_inv, convert_points_to_homogeneous(point_2d) * depth)

    def _projection_matrix(self) -> torch.Tensor:
        r"""Return the 4x4 projection built from the intrinsics' top-left 3x3 block.

        Both :meth:`project` and :meth:`unproject` go through this matrix, so they agree on
        which intrinsics are usable: a 3x3 ``K`` zero-padded to 4x4 round-trips exactly like its
        homogeneous embedding, instead of projecting while ``unproject`` raises on the singular
        4x4 product. For a homogeneous embedding the result matches ``intrinsics @ extrinsics``
        bit for bit on CPU, and to the dtype's rounding on MPS.
        """
        K = self._intrinsics[..., :3, :3]
        E = self._extrinsics
        P = torch.zeros(*K.shape[:-2], 4, 4, device=K.device, dtype=K.dtype)
        P[..., :3, :3] = K @ E[..., :3, :3]
        P[..., :3, 3:] = K @ E[..., :3, 3:]
        P[..., 3, 3] = 1.0
        return P

    # NOTE: just for test. Decide if we keep it.
    @classmethod
    def from_parameters(
        self,
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        height: int,
        width: int,
        tx: torch.Tensor,
        ty: torch.Tensor,
        tz: torch.Tensor,
        batch_size: int,
        device: Union[str, torch.device, None],
        dtype: torch.dtype,
    ) -> "PinholeCamera":
        r"""Construct a batched pinhole camera from scalar parameter tensors.

        See the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

        This helper allocates batched :math:`4 \times 4` intrinsic and
        extrinsic matrices, fills focal lengths/principal point/translation,
        and wraps them into a :class:`PinholeCamera` instance.

        Convention:
            - every parameter, including ``height`` and ``width``, is broadcast over ``batch_size``.
            - ``height`` and ``width`` are stored as floating-point tensors of the requested ``dtype``, so the
              camera it builds can be scaled in place.

        Args:
            fx: Horizontal focal length per batch element.
            fy: Vertical focal length per batch element.
            cx: Principal point x-coordinate per batch element.
            cy: Principal point y-coordinate per batch element.
            height: Image height in pixels.
            width: Image width in pixels.
            tx: Camera translation along x per batch element.
            ty: Camera translation along y per batch element.
            tz: Camera translation along z per batch element.
            batch_size: Number of cameras in the batch.
            device: Target device for the created tensors.
            dtype: Target floating-point dtype for the created tensors.

        Returns:
            Batched :class:`PinholeCamera` configured from the provided
            intrinsic and translation parameters.
        """
        # create the camera matrix
        intrinsics = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
        intrinsics[..., 0, 0] += fx
        intrinsics[..., 1, 1] += fy
        intrinsics[..., 0, 2] += cx
        intrinsics[..., 1, 2] += cy
        intrinsics[..., 2, 2] += 1.0
        intrinsics[..., 3, 3] += 1.0
        # create the pose matrix
        extrinsics = torch.eye(4, device=device, dtype=dtype).repeat(batch_size, 1, 1)
        extrinsics[..., 0, -1] += tx
        extrinsics[..., 1, -1] += ty
        extrinsics[..., 2, -1] += tz
        # create image height and width, one entry per batch element
        height_tmp = torch.full((batch_size,), height, device=device, dtype=dtype)
        width_tmp = torch.full((batch_size,), width, device=device, dtype=dtype)
        return self(intrinsics, extrinsics, height_tmp, width_tmp)


class PinholeCamerasList(PinholeCamera):
    r"""Class that represents a list of pinhole cameras.

    The class inherits from :class:`~kornia.PinholeCamera` meaning that
    it will keep the same class properties but with an extra dimension.

    .. note::
        The underlying data torch.Tensor will be stacked in the first dimension.
        That's it, given a list of two camera instances, the intrinsics torch.Tensor
        will have a shape ``(B, N, 4, 4)`` where :math:`B` is the batch
        size and :math:`N` is the numbers of cameras (in this case two).

    Args:
        pinholes_list: a python tuple or list containing a set of `PinholeCamera` instances.

    """

    def __init__(self, pinholes_list: Iterable[PinholeCamera]) -> None:
        self._initialize_parameters(pinholes_list)

    @staticmethod
    def _check_valid_params(data: torch.Tensor, data_name: str) -> bool:
        if len(data.shape) != 4 or data.shape[-2:] != (4, 4):
            raise ValueError(f"Argument {data_name} shape must be in the following shape BxNx4x4. Got {data.shape}")
        return True

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
        # torch.cat and set members. We will assume BxNx4x4
        self.height: torch.Tensor = torch.stack(height, dim=1)
        self.width: torch.Tensor = torch.stack(width, dim=1)
        self._intrinsics: torch.Tensor = torch.stack(intrinsics, dim=1)
        self._extrinsics: torch.Tensor = torch.stack(extrinsics, dim=1)
        return self

    def clone(self) -> "PinholeCamerasList":
        """Return a copy of the list, with each camera cloned."""
        return PinholeCamerasList([self.get_pinhole(index).clone() for index in range(self.num_cameras)])

    def scale(self, scale_factor: torch.Tensor) -> "PinholeCamerasList":
        """Scale the cameras by a vector factor along the batch dimension, returning a new list."""
        return self.clone().scale_(scale_factor)

    def scale_(self, scale_factor: Union[float, torch.Tensor]) -> "PinholeCamerasList":
        r"""Scale the cameras in place, applying a vector factor along the batch dimension."""
        if isinstance(scale_factor, torch.Tensor) and scale_factor.ndim == 1:
            scale_factor = scale_factor[..., None]
        self.intrinsics[..., 0, 0] *= scale_factor
        self.intrinsics[..., 1, 1] *= scale_factor
        self.intrinsics[..., 0, 2] *= scale_factor
        self.intrinsics[..., 1, 2] *= scale_factor
        self.height = self.height * scale_factor
        self.width = self.width * scale_factor
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
        height: torch.Tensor = self.height[..., idx]
        width: torch.Tensor = self.width[..., idx]
        intrinsics: torch.Tensor = self.intrinsics[:, idx]
        extrinsics: torch.Tensor = self.extrinsics[:, idx]
        return PinholeCamera(intrinsics, extrinsics, height, width)


# based on:
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L26


def pixel2cam(depth: torch.Tensor, intrinsics_inv: torch.Tensor, pixel_coords: torch.Tensor) -> torch.Tensor:
    r"""Unproject a depth map to camera-frame points through a 4x4 inverse intrinsics matrix.

    ``points = pixel2cam(depth, intrinsics_inv, pixel_coords)`` takes a ``(B, 1, H, W)`` depth map, the
    ``(B, 4, 4)`` inverse intrinsics that :meth:`PinholeCamera.intrinsics_inverse` returns and the homogeneous
    pixel grid ``(B, H, W, 3)`` of ``(u, v, 1)``, and returns the camera-frame point ``depth * K^-1 (u, v, 1)`` of
    every pixel as ``(B, H, W, 3)``. It is the first half of the dense warp of
    :class:`~kornia.geometry.depth.DepthWarper`; :func:`cam2pixel` is the second. For a ``3x3`` ``K``, use
    :func:`~kornia.geometry.depth.depth_to_3d_v2`, which builds the pixel grid itself.

    Convention:
        - ``intrinsics_inv`` is a ``(B, 4, 4)`` inverse calibration matrix (the
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera` layout, not a ``3x3`` ``K``), and ``depth`` is
          the camera-frame ``z`` at each pixel of the ``(u, v, 1)`` grid. Pixel centres are integers; see the
          Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - a ``depth`` that is not ``(B, 1, H, W)``, an ``intrinsics_inv`` that is not ``(B, 4, 4)`` or a
          ``pixel_coords`` that is not ``(B, H, W, 3)`` raises :class:`ValueError`. That ``pixel_coords`` matches
          the ``H`` and ``W`` of ``depth`` is not checked up front: the two broadcast, so a mismatch raises a
          broadcasting ``RuntimeError`` unless one of the sizes is 1.
        - the batch of ``intrinsics_inv`` is that of ``pixel_coords`` or 1, and ``depth`` broadcasts against the
          result: one camera and one grid serve a batch of depth maps.

    Args:
        depth: the source depth maps. Shape must be ``(B, 1, H, W)``.
        intrinsics_inv: the inverse intrinsics camera matrix. Shape must be ``(B, 4, 4)``.
        pixel_coords: the grid with (u, v, 1) pixel coordinates. Shape must be ``(B, H, W, 3)``.

    Returns:
        torch.Tensor of shape ``(B, H, W, 3)`` with (x, y, z) cam coordinates.

    See Also:
        - :func:`cam2pixel`: the second half of the warp, camera-frame points to the pixels of a camera.
        - :class:`PinholeCamera`: the camera model whose ``(B, 4, 4)`` matrices this function takes.
        - :func:`~kornia.geometry.depth.depth_to_3d_v2`: the same points from a ``3x3`` ``K``, without a grid
          to pass.
        - :func:`~kornia.geometry.camera.perspective.unproject_points`: selected pixels at given depths, with a
          ``3x3`` ``K``.

    """
    if not (len(depth.shape) == 4 and depth.shape[1] == 1):
        raise ValueError(f"Input depth has to be in the shape of Bx1xHxW. Got {depth.shape}")
    if not (len(intrinsics_inv.shape) == 3 and intrinsics_inv.shape[-2:] == (4, 4)):
        raise ValueError(f"Input intrinsics_inv has to be in the shape of Bx4x4. Got {intrinsics_inv.shape}")
    if not (len(pixel_coords.shape) == 4 and pixel_coords.shape[3] == 3):
        raise ValueError(f"Input pixel_coords has to be in the shape of BxHxWx3. Got {pixel_coords.shape}")
    cam_coords: torch.Tensor = transform_points(intrinsics_inv[:, None], pixel_coords)
    return cam_coords * depth.permute(0, 2, 3, 1)


# based on
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43


def cam2pixel(cam_coords_src: torch.Tensor, dst_proj_src: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Project a map of camera-frame points to pixels through a 4x4 projection matrix.

    ``uv = cam2pixel(cam_coords_src, dst_proj_src)`` takes a ``(B, H, W, 3)`` map of points in the source camera
    frame and the ``(B, 4, 4)`` projection ``dst_proj_src``, which moves a source-frame point into the destination
    camera frame and applies the destination intrinsics, and returns their ``(u, v)`` pixel coordinates in the
    destination camera, ``(B, H, W, 2)``. It is the second half of the dense warp of
    :class:`~kornia.geometry.depth.DepthWarper`, after :func:`pixel2cam`. For points already in a camera's frame
    and a ``3x3`` ``K``, use :func:`~kornia.geometry.camera.perspective.project_points`.

    Convention:
        - ``dst_proj_src`` is a ``(B, 4, 4)`` projection matrix (the
          :class:`~kornia.geometry.camera.pinhole.PinholeCamera` layout, not a ``3x3`` ``K``), and the result is
          ``(u, v)`` pixel coordinates in the destination frame, with integer pixel centres; see the Convention
          block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.
        - the batch of ``dst_proj_src`` is that of ``cam_coords_src`` or 1. A point behind the camera is divided
          like any other and is not flagged.

    .. warning::
        The perspective division is ``x / (z + eps)`` rather than a guarded divide, so ``z = 0`` gives
        ``x / eps``, huge for a nonzero ``x`` (``inf`` where ``eps`` underflows), and ``eps`` biases every small
        depth:
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_.

    Args:
        cam_coords_src: (x, y, z) coordinates defined in the first camera coordinates system. Shape must be
            ``(B, H, W, 3)``.
        dst_proj_src: the projection matrix between the
          reference and the non reference camera frame. Shape must be ``(B, 4, 4)``.
        eps: small value to avoid division by zero error.

    Returns:
        torch.Tensor of shape ``(B, H, W, 2)`` with (u, v) pixel coordinates.

    See Also:
        - :func:`pixel2cam`: the first half of the warp, a depth map to camera-frame points.
        - :class:`PinholeCamera`: the camera model whose ``(B, 4, 4)`` matrices compose ``dst_proj_src``.
        - :func:`~kornia.geometry.camera.perspective.project_points`: camera-frame points to pixels with a
          ``3x3`` ``K``, skipping the divide at ``z = 0`` instead of adding ``eps``.

    """
    if not (len(cam_coords_src.shape) == 4 and cam_coords_src.shape[3] == 3):
        raise ValueError(f"Input cam_coords_src has to be in the shape of BxHxWx3. Got {cam_coords_src.shape}")
    if not (len(dst_proj_src.shape) == 3 and dst_proj_src.shape[-2:] == (4, 4)):
        raise ValueError(f"Input dst_proj_src has to be in the shape of Bx4x4. Got {dst_proj_src.shape}")
    # apply projection matrix to points
    point_coords: torch.Tensor = transform_points(dst_proj_src[:, None], cam_coords_src)
    x_coord: torch.Tensor = point_coords[..., 0]
    y_coord: torch.Tensor = point_coords[..., 1]
    z_coord: torch.Tensor = point_coords[..., 2]

    # compute pixel coordinates
    u_coord: torch.Tensor = x_coord / (z_coord + eps)
    v_coord: torch.Tensor = y_coord / (z_coord + eps)

    # torch.stack and return the coordinates, that's the actual flow
    pixel_coords_dst: torch.Tensor = torch.stack([u_coord, v_coord], dim=-1)
    return pixel_coords_dst  # BxHxWx2
