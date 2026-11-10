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

# adapted from: https://github.com/strasdat/Sophus/blob/sophus2/cpp/sophus/sensor/camera_model.h
from __future__ import annotations

from enum import Enum
from typing import Any, Union

import torch

from kornia.geometry.vector import Vector2, Vector3
from kornia.image import ImageSize
from kornia.sensors.camera.distortion_model import AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform
from kornia.sensors.camera.projection_model import OrthographicProjection, Z1Projection


class CameraModelType(Enum):
    """Name a built-in camera model for :class:`CameraModel`: pinhole, Brown-Conrady, fisheye or orthographic.

    ``CameraModel(image_size, CameraModelType.PINHOLE, params)`` builds the model a member names, and the member fixes
    the length of ``params``:

    - ``PINHOLE``: :class:`PinholeModel`, perspective projection with no lens distortion, ``[fx, fy, cx, cy]``.
    - ``BROWN_CONRADY``: :class:`BrownConradyModel`, OpenCV's radial and tangential lens distortion, 12 values.
    - ``KANNALA_BRANDT_K3``: :class:`KannalaBrandtK3`, the fisheye model whose radius is a 9th-order odd polynomial
      of the ray angle, 8 values.
    - ``ORTHOGRAPHIC``: :class:`Orthographic`, parallel projection that drops ``z``, ``[fx, fy, cx, cy]``.

    See Also:
        - :class:`CameraModel`: builds the model a member names.
        - :class:`CameraModelBase`: composes a custom model from a distortion and a projection.

    """

    PINHOLE = 0
    BROWN_CONRADY = 1
    KANNALA_BRANDT_K3 = 2
    ORTHOGRAPHIC = 3


def get_model_from_type(
    model_type: CameraModelType, image_size: ImageSize, params: torch.Tensor
) -> CameraModelVariants:
    """Get camera model from model type."""
    if model_type == CameraModelType.PINHOLE:
        return PinholeModel(image_size, params)
    if model_type == CameraModelType.BROWN_CONRADY:
        return BrownConradyModel(image_size, params)
    if model_type == CameraModelType.KANNALA_BRANDT_K3:
        return KannalaBrandtK3(image_size, params)
    if model_type == CameraModelType.ORTHOGRAPHIC:
        return Orthographic(image_size, params)
    raise ValueError("Invalid Camera Model Type")


CameraDistortionType = Union[AffineTransform, BrownConradyTransform, KannalaBrandtK3Transform]
CameraProjectionType = Union[Z1Projection, OrthographicProjection]

# The number of parameters each distortion model reads, which is what fixes the
# trailing dimension of ``params``. The typed constructors below each spell the
# same number out; this is the one place that maps it to the distortion, so the
# base class can enforce the shape its own docstring documents.
_PARAMS_LEN_FOR_DISTORTION: dict[type, int] = {
    AffineTransform: 4,
    BrownConradyTransform: 12,
    KannalaBrandtK3Transform: 8,
}


def _validate_params(distortion: CameraDistortionType, params: torch.Tensor) -> None:
    """Check ``params`` against the shape :class:`CameraModelBase` documents.

    ``CameraModelBase`` is public and its docstring states a shape, but only the
    typed subclasses checked it. Constructing the base directly with a short
    vector deferred the failure to an ``IndexError`` inside the distortion model,
    naming neither ``params`` nor the camera; a rank-3 ``(B, 1, N)`` tensor --
    the shape the subclasses reject, because there is no multi-camera form --
    was accepted and projected to a silently wrong ``(1, 1, 2)`` result.

    Raises:
        ValueError: if ``params`` is not of shape ``(N,)`` or ``(B, N)``,
            with ``N`` the length the distortion model reads.

    """
    if params.ndim not in (1, 2):
        raise ValueError(f"params must be of rank 1 or 2, of shape (B, N) or (N,); got shape {tuple(params.shape)}")
    expected = _PARAMS_LEN_FOR_DISTORTION.get(type(distortion))
    if expected is not None and params.shape[-1] != expected:
        raise ValueError(
            f"params must be of shape (B, {expected}) or ({expected},) for "
            f"{type(distortion).__name__}; got shape {tuple(params.shape)}"
        )


class CameraModelBase:
    """Compose a camera model from a distortion and a projection, to define a custom model.

    ``cam = CameraModelBase(distortion, projection, image_size, params)`` takes a distortion from
    ``kornia.sensors.camera.distortion_model``, a projection from ``kornia.sensors.camera.projection_model`` and the
    flat ``params`` the distortion reads. ``cam.project(points)`` maps a :class:`~kornia.geometry.vector.Vector3` of
    camera-frame points to a :class:`~kornia.geometry.vector.Vector2` of pixels, and ``cam.unproject(pixels, depth)``
    maps pixels at camera-frame depths back to points. For a built-in model, use :class:`CameraModel` or one of the
    subclasses.

    Convention:
        - :meth:`project` is ``distortion.distort(params, projection.project(points))`` and :meth:`unproject`
          ``projection.unproject(distortion.undistort(params, pixels), depth)``. ``depth`` is the camera-frame ``z``,
          shaped ``(*,)`` where :func:`~kornia.geometry.camera.perspective.unproject_points` takes ``(*, 1)``. Pixel
          centers are integers (see :class:`~kornia.geometry.camera.pinhole.PinholeCamera`).
        - points and pixels are ``Vector3`` and ``Vector2``: wrap a tensor as ``Vector3(points)``. This type system is
          kept separate from :doc:`kornia.geometry.camera </geometry.camera>` by design
          (`#4274 <https://github.com/kornia/kornia/issues/4274>`_).
        - :meth:`matrix` and its alias :meth:`K` return the ``(*, 3, 3)`` ``K``, not the ``(B, 4, 4)``
          ``intrinsics`` that :class:`~kornia.geometry.camera.pinhole.PinholeCamera` stores.

    .. warning::
        Batched ``(B, 4)`` and ``(B, 8)`` parameters apply along the **last point axis**, not the camera axis: with
        ``(B, N, 3)`` points and ``B > 1`` they silently pair with the wrong points when ``B == N``,
        :meth:`project` broadcasts to a ``(B, B, 2)`` result when ``N == 1``, and other ``N`` raise, whereas the
        geometry functions apply them per camera:
        `#4274 <https://github.com/kornia/kornia/issues/4274>`_. Batched ``(B, 12)`` Brown-Conrady parameters pair
        camera ``b`` with ``points[b]``. :meth:`project` divides by ``z`` with no guard, so ``z = 0`` gives
        infinities where :func:`~kornia.geometry.camera.perspective.project_points` returns finite pixels:
        `#4267 <https://github.com/kornia/kornia/issues/4267>`_.

    Args:
        distortion: the map from the image-plane points ``projection`` returns to pixels, and back.
        projection: the map from camera-frame points to the image plane, and back.
        image_size: :class:`~kornia.image.ImageSize` of the image, returned by :attr:`image_size`, :attr:`height` and
            :attr:`width`; :meth:`project` and :meth:`unproject` do not read it.
        params: camera parameters of shape ``(N,)`` or ``(B, N)`` that start with ``[fx, fy, cx, cy]``. ``N`` is 4 for
            ``AffineTransform`` (the pinhole and orthographic models), 12 for ``BrownConradyTransform`` and 8 for
            ``KannalaBrandtK3Transform``.

    Raises:
        ValueError: if ``params`` does not have that shape.

    See Also:
        - :class:`CameraModel`: a built-in model chosen by a :class:`CameraModelType`.
        - :class:`~kornia.geometry.camera.pinhole.PinholeCamera`: plain tensors, ``(B, 4, 4)`` intrinsics next to
          world-to-camera extrinsics, and batched cameras applied per camera.

    Example:
        >>> params = torch.Tensor([328., 328., 320., 240.])
        >>> cam = CameraModelBase(AffineTransform(), Z1Projection(), ImageSize(480, 640), params)
        >>> cam.params
        tensor([328., 328., 320., 240.])

    """

    def __init__(
        self,
        distortion: CameraDistortionType,
        projection: CameraProjectionType,
        image_size: ImageSize,
        params: torch.Tensor,
    ) -> None:
        _validate_params(distortion, params)
        self.distortion = distortion
        self.projection = projection
        self._image_size = image_size
        self._height = image_size.height
        self._width = image_size.width
        self._params = params

    @property
    def image_size(self) -> ImageSize:
        """Returns the image size of the camera model."""
        return self._image_size

    @property
    def height(self) -> int | torch.Tensor:
        """Returns the height of the image."""
        return self._height

    @property
    def width(self) -> int | torch.Tensor:
        """Returns the width of the image."""
        return self._width

    @property
    def params(self) -> torch.Tensor:
        """Returns the camera parameters."""
        return self._params

    @property
    def fx(self) -> torch.Tensor:
        """Returns the focal length in x direction."""
        return self._params[..., 0]

    @property
    def fy(self) -> torch.Tensor:
        """Returns the focal length in y direction."""
        return self._params[..., 1]

    @property
    def cx(self) -> torch.Tensor:
        """Returns the principal point in x direction."""
        return self._params[..., 2]

    @property
    def cy(self) -> torch.Tensor:
        """Returns the principal point in y direction."""
        return self._params[..., 3]

    def matrix(self) -> torch.Tensor:
        """Return the ``(*, 3, 3)`` camera matrix ``K`` built from ``fx``, ``fy``, ``cx`` and ``cy``.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        """
        z = torch.zeros_like(self.fx)
        row1 = torch.stack((self.fx, z, self.cx), -1)
        row2 = torch.stack((z, self.fy, self.cy), -1)
        row3 = torch.stack((z, z, z), -1)
        K = torch.stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    def K(self) -> torch.Tensor:
        """Return the ``(*, 3, 3)`` camera matrix ``K``; an alias of :meth:`matrix`.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        """
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        """Project camera-frame points to pixels.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        Args:
            points: ``Vector3`` of camera-frame points.

        Returns:
            ``Vector2`` of the pixels.

        Example:
            >>> points = Vector3(torch.Tensor([1.0, 1.0, 1.0]))
            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
            >>> cam.project(points)
            x: 648.0
            y: 568.0

        """
        return self.distortion.distort(self.params, self.projection.project(points))

    def unproject(self, points: Vector2, depth: torch.Tensor) -> Vector3:
        """Unproject pixels at camera-frame depths to camera-frame points.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        Args:
            points: ``Vector2`` of pixels.
            depth: camera-frame ``z`` of each point, ``(*,)``.

        Returns:
            ``Vector3`` of the camera-frame points.

        Example:
            >>> points = Vector2(torch.Tensor([1.0, 1.0]))
            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
            >>> cam.unproject(points, torch.Tensor([1.0]))
            x: tensor([-0.9726])
            y: tensor([-0.7287])
            z: tensor([1.])

        """
        return self.projection.unproject(self.distortion.undistort(self.params, points), depth)


class PinholeModel(CameraModelBase):
    """Model a pinhole camera: perspective projection with ``[fx, fy, cx, cy]`` and no lens distortion.

    ``cam = PinholeModel(image_size, params)`` takes the ``(4,)`` or ``(B, 4)`` intrinsics ``[fx, fy, cx, cy]``.
    :meth:`~CameraModelBase.project` maps a camera-frame point ``(x, y, z)`` to the pixel
    ``(fx * x / z + cx, fy * y / z + cy)``, and :meth:`~CameraModelBase.unproject` inverts it in closed form. It is
    :class:`CameraModelBase` with :class:`~kornia.sensors.camera.distortion_model.AffineTransform` and
    :class:`~kornia.sensors.camera.projection_model.Z1Projection`.

    See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`, and
    :doc:`camera and world conventions </get-started/camera-conventions>` for the camera frame, depth, pixel-center
    and intrinsics conventions.

    Args:
        image_size: :class:`~kornia.image.ImageSize` of the image.
        params: ``(4,)`` or ``(B, 4)`` intrinsics ``[fx, fy, cx, cy]``; another length, or a rank
            above 2, raises ``ValueError``.

    See Also:
        - :class:`BrownConradyModel`: the same projection, with OpenCV's radial and tangential lens distortion.
        - :class:`KannalaBrandtK3`: a fisheye model, with the radius a polynomial of the ray angle.
        - :class:`Orthographic`: the same intrinsics after dropping ``z`` instead of dividing by it.

    Example:
        >>> cam = PinholeModel(ImageSize(480, 640), torch.Tensor([328., 328., 320., 240.]))
        >>> cam.project(Vector3(torch.Tensor([1.0, 1.0, 1.0])))
        x: 648.0
        y: 568.0

    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError("params must be of shape (B, 4) for PINHOLE Camera")
        super().__init__(AffineTransform(), Z1Projection(), image_size, params)

    def matrix(self) -> torch.Tensor:
        r"""Return the camera matrix.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        The matrix is of the form:

        .. math::
            \begin{bmatrix} fx & 0 & cx \\
            0 & fy & cy \\
            0 & 0 & 1\end{bmatrix}

        Example:
            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([1.0, 2.0, 3.0, 4.0]))
            >>> cam.matrix()
            tensor([[1., 0., 3.],
                    [0., 2., 4.],
                    [0., 0., 1.]])

        """
        z = torch.zeros_like(self.fx)
        row1 = torch.stack((self.fx, z, self.cx), -1)
        row2 = torch.stack((z, self.fy, self.cy), -1)
        row3 = torch.stack((z, z, z), -1)
        K = torch.stack((row1, row2, row3), -2)
        K[..., -1, -1] = 1.0
        return K

    def scale(self, scale_factor: torch.Tensor) -> PinholeModel:
        """Return a new pinhole model with the intrinsics and the image size multiplied by a factor.

        Convention:
            - returns a **new** model whose focal lengths, principal point and image size are multiplied by
              ``scale_factor``: ``fx' = s * fx`` and ``cx' = s * cx``, as
              :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale` does.

        .. warning::
            ``cx' = s * cx`` disagrees with kornia's integer pixel centres, and a tensor ``scale_factor``
            rebuilds ``image_size`` with 0-dim tensors where the constructor took python integers:
            `#4263 <https://github.com/kornia/kornia/issues/4263>`_.

        Args:
            scale_factor: Scale factor to scale the camera model.

        Returns:
            Scaled camera model.

        Example:
            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
            >>> cam_scaled = cam.scale(2)
            >>> cam_scaled.params
            tensor([656., 656., 640., 480.])

        """
        fx = self.fx * scale_factor
        fy = self.fy * scale_factor
        cx = self.cx * scale_factor
        cy = self.cy * scale_factor
        params = torch.stack((fx, fy, cx, cy), -1)
        image_size = ImageSize(self.image_size.height * scale_factor, self.image_size.width * scale_factor)
        return PinholeModel(image_size, params)


class BrownConradyModel(CameraModelBase):
    """Model a camera with OpenCV's Brown-Conrady radial and tangential lens distortion.

    ``cam = BrownConradyModel(image_size, params)`` takes the ``(12,)`` or ``(B, 12)`` vector
    ``[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]``: the intrinsics, then OpenCV's eight distortion coefficients.
    :meth:`~CameraModelBase.project` divides camera-frame points by ``z`` and distorts them to pixels, and
    :meth:`~CameraModelBase.unproject` removes the distortion by fixed-point iteration. It is :class:`CameraModelBase`
    with :class:`~kornia.sensors.camera.distortion_model.BrownConradyTransform` and
    :class:`~kornia.sensors.camera.projection_model.Z1Projection`.

    See :doc:`camera and world conventions </get-started/camera-conventions>` for shared intrinsics,
    pixel-center and camera-frame conventions.

    Args:
        image_size: :class:`~kornia.image.ImageSize` of the image.
        params: ``(12,)`` or ``(B, 12)`` parameters in the order above; another length, or a rank
            above 2, raises ``ValueError``.

    See Also:
        - :class:`PinholeModel`: the same projection with no lens distortion, in closed form both ways.
        - :class:`KannalaBrandtK3`: a fisheye model, whose radius is a polynomial of the ray angle instead of the
          normalized radius.

    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        if params.shape[-1] != 12 or len(params.shape) > 2:
            raise ValueError("params must be of shape (B, 12) for BROWN_CONRADY Camera")
        super().__init__(BrownConradyTransform(), Z1Projection(), image_size, params)


class KannalaBrandtK3(CameraModelBase):
    r"""Model a fisheye camera with the Kannala-Brandt K3 distortion of the ray angle.

    ``cam = KannalaBrandtK3(image_size, params)`` takes the ``(8,)`` or ``(B, 8)`` vector
    ``[fx, fy, cx, cy, k0, k1, k2, k3]``. :meth:`~CameraModelBase.project` maps the angle :math:`\theta` between a
    point's ray and the optical axis to the normalized radius
    :math:`\theta (1 + k_0 \theta^2 + k_1 \theta^4 + k_2 \theta^6 + k_3 \theta^8)`, and
    :meth:`~CameraModelBase.unproject` inverts it iteratively. It is :class:`CameraModelBase` with
    :class:`~kornia.sensors.camera.distortion_model.KannalaBrandtK3Transform` and
    :class:`~kornia.sensors.camera.projection_model.Z1Projection`.

    See :doc:`camera and world conventions </get-started/camera-conventions>` for shared intrinsics,
    pixel-center and camera-frame conventions.

    Args:
        image_size: :class:`~kornia.image.ImageSize` of the image.
        params: ``(8,)`` or ``(B, 8)`` parameters in the order above; another length, or a rank
            above 2, raises ``ValueError``.

    See Also:
        - :class:`BrownConradyModel`: OpenCV's distortion of the normalized radius, with tangential terms.
        - :class:`PinholeModel`: the same projection with no lens distortion.

    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        if params.shape[-1] != 8 or len(params.shape) > 2:
            raise ValueError("params must be of shape B, 8 for KANNALA_BRANDT_K3 Camera")
        super().__init__(KannalaBrandtK3Transform(), Z1Projection(), image_size, params)


class Orthographic(CameraModelBase):
    """Model an orthographic camera: parallel projection that drops ``z``, then ``[fx, fy, cx, cy]``.

    ``cam = Orthographic(image_size, params)`` takes the ``(4,)`` or ``(B, 4)`` intrinsics ``[fx, fy, cx, cy]``.
    :meth:`~CameraModelBase.project` maps a camera-frame point ``(x, y, z)`` to the pixel ``(fx * x + cx, fy * y + cy)``
    whatever its ``z``, and :meth:`~CameraModelBase.unproject` returns ``((u - cx) / fx, (v - cy) / fy, depth)``. It is
    :class:`CameraModelBase` with :class:`~kornia.sensors.camera.distortion_model.AffineTransform` and
    :class:`~kornia.sensors.camera.projection_model.OrthographicProjection`.

    See :doc:`camera and world conventions </get-started/camera-conventions>` for shared intrinsics,
    pixel-center and camera-frame conventions.

    Args:
        image_size: :class:`~kornia.image.ImageSize` of the image.
        params: ``(4,)`` or ``(B, 4)`` intrinsics ``[fx, fy, cx, cy]``; another length, or a rank
            above 2, raises ``ValueError``.

    See Also:
        - :class:`PinholeModel`: the same intrinsics after dividing by ``z``, so distant objects shrink.

    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError("params must be of shape B, 4 for ORTHOGRAPHIC Camera")
        super().__init__(AffineTransform(), OrthographicProjection(), image_size, params)


CameraModelVariants = Union[PinholeModel, BrownConradyModel, KannalaBrandtK3, Orthographic]


class CameraModel:
    """Build a built-in camera model from a :class:`CameraModelType` and a flat parameter vector.

    ``cam = CameraModel(image_size, model_type, params)`` builds the model that ``model_type`` names and forwards every
    attribute to it: ``cam.project(points)`` and ``cam.unproject(pixels, depth)`` call that model's methods. ``cam`` is
    a wrapper, not a :class:`CameraModelBase` instance; ``cam.scale(s)`` of a pinhole model returns a
    :class:`PinholeModel`.

    See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`, and
    :doc:`camera and world conventions </get-started/camera-conventions>` for the camera frame, depth, pixel-center
    and intrinsics conventions.

    Args:
        image_size: :class:`~kornia.image.ImageSize` of the image.
        model_type: the model to build; a value that is not a :class:`CameraModelType` raises ``ValueError``.
        params: ``(N,)`` or ``(B, N)`` parameters, with ``N`` fixed by ``model_type`` (see :class:`CameraModelType`).

    See Also:
        - :class:`CameraModelType`: the models it builds, and the length of ``params`` for each.
        - :class:`CameraModelBase`: composes a custom model from a distortion and a projection.

    Example:
        >>> # Pinhole Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
        >>> # Brown Conrady Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.BROWN_CONRADY, torch.Tensor([1.0, 1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Kannala Brandt K3 Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.KANNALA_BRANDT_K3, torch.Tensor([1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Orthographic Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.ORTHOGRAPHIC, torch.Tensor([328., 328., 320., 240.]))
        >>> cam.params
        tensor([328., 328., 320., 240.])

    """

    def __init__(self, image_size: ImageSize, model_type: CameraModelType, params: torch.Tensor) -> None:
        self._model = get_model_from_type(model_type, image_size, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"CameraModel({self.image_size}, {self._model.__class__.__name__}, {self.params})"
