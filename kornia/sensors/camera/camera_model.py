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
    """Represent the type of camera projection and distortion model.

    Supported types:
        - PINHOLE: Standard perspective projection with no distortion.
        - BROWN_CONRADY: Standard radial and tangential distortion model, often used for wide-angle lenses.
        - KANNALA_BRANDT_K3: Fisheye distortion model using a 9th-order polynomial for equidistant projection.
        - ORTHOGRAPHIC: Parallel projection where rays are perpendicular to the image plane, with no perspective effect.
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
    elif model_type == CameraModelType.BROWN_CONRADY:
        return BrownConradyModel(image_size, params)
    elif model_type == CameraModelType.KANNALA_BRANDT_K3:
        return KannalaBrandtK3(image_size, params)
    elif model_type == CameraModelType.ORTHOGRAPHIC:
        return Orthographic(image_size, params)
    else:
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
        ValueError: if ``params`` is not of shape :math:`(N,)` or :math:`(B, N)`,
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
    r"""Base class to represent camera models based on distortion and projection types.

    Distortion is of 3 types:
        - Affine, implemented by :class:`~kornia.sensors.camera.distortion_model.AffineTransform`
        - Brown Conrady, a placeholder that raises ``NotImplementedError``
        - Kannala Brandt K3, a placeholder that raises ``NotImplementedError``
    Projection is of 2 types:
        - Z1, implemented by :class:`~kornia.sensors.camera.projection_model.Z1Projection`
        - Orthographic, a placeholder that raises ``NotImplementedError``

    Convention:
        - the API is ``Vector``-typed: :meth:`project` takes a ``Vector3`` and returns a ``Vector2``, and
          :meth:`unproject` takes a ``Vector2`` and returns a ``Vector3``. A raw :class:`torch.Tensor` is not
          accepted -- reading a coordinate off it raises ``AttributeError``.
        - ``params`` is a flat parameter vector whose length is fixed by the :class:`CameraModelType`:
          ``[fx, fy, cx, cy]`` for ``PINHOLE`` and ``ORTHOGRAPHIC``, 12 parameters for ``BROWN_CONRADY`` and
          8 for ``KANNALA_BRANDT_K3``, laid out as each constructor documents. That length is enforced by
          the typed constructors -- :class:`CameraModel` and the :class:`PinholeModel`,
          :class:`BrownConradyModel`, :class:`KannalaBrandtK3` and :class:`Orthographic` subclasses -- which
          accept an unbatched ``(N,)`` vector and a batched :math:`(B, N)` one and raise ``ValueError`` for
          another length or a rank above 2. ``CameraModelBase.__init__`` applies the same check on the direct
          construction path, reading the length off the distortion type -- 4 for
          :class:`~kornia.sensors.camera.distortion_model.AffineTransform`, 12 for ``BrownConradyTransform``
          and 8 for ``KannalaBrandtK3Transform``.
        - :meth:`matrix` and its alias :meth:`K` return the :math:`(*, 3, 3)` intrinsics
          ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, carrying the batch axis of ``params`` -- not the
          :math:`(B, 4, 4)` ``intrinsics`` that :class:`~kornia.geometry.camera.pinhole.PinholeCamera`
          stores, whose :attr:`~kornia.geometry.camera.pinhole.PinholeCamera.camera_matrix` property returns
          the :math:`(B, 3, 3)` block of it. :class:`PinholeModel` implements it; ``CameraModelBase.matrix``
          itself raises ``NotImplementedError``.
        - :meth:`project` is ``self.distortion.distort(self.params, self.projection.project(points))`` and
          :meth:`unproject` the reverse,
          ``self.projection.unproject(self.distortion.undistort(self.params, points), depth)``. ``depth`` is
          the camera-frame ``z``: :class:`~kornia.sensors.camera.projection_model.Z1Projection` multiplies
          the :math:`z = 1` point by it, so the third coordinate of the result is the ``depth`` that was
          passed in, and not a Euclidean ray length.
        - with shared intrinsics ``params.shape == (4,)``, or one point per camera with ``params`` of shape
          ``(B, 4)`` and points of shape ``(B, 3)`` / ``(B, 2)``, the Pinhole path uses the same mathematical
          camera mapping as :doc:`kornia.geometry.camera </geometry.camera>` with ``K`` built from the same
          ``[fx, fy, cx, cy]``.
          However, :meth:`project` divides directly by ``z``, whereas
          :func:`~kornia.geometry.camera.perspective.project_points` multiplies by its reciprocal and skips
          the divide when ``abs(z) <= 1e-8`` (compared in the working dtype). Results can differ by rounding
          away from that threshold and differ substantially at or below it: ``[1, 2, 0]`` yields infinities
          here but finite pixels there. See `#4267 <https://github.com/kornia/kornia/issues/4267>`_.
          :meth:`unproject` corresponds to :func:`~kornia.geometry.camera.perspective.unproject_points`
          with ``normalize=False`` and depth shaped as ``(*, 1)`` there instead of ``(*,)`` here.
        - for point clouds shaped ``(B, N, 3)`` / ``(B, N, 2)``, batched ``(B, 4)`` intrinsics broadcast
          differently: this API applies each ``(B,)`` intrinsic component directly to ``(B, N)`` coordinates,
          aligning it with the point axis. The geometry functions insert a singleton point axis and apply
          intrinsics along the camera batch axis. When ``B == N > 1``, both APIs run but associate the
          intrinsics with different points; with ``B = 2, N = 3``, both :meth:`project` and :meth:`unproject`
          here raise ``RuntimeError`` while the geometry functions support those shapes. Changing only the
          point container and depth shape is therefore insufficient for batched point clouds.
        - pixels use the integer-centre grid described in
          the Convention block on :class:`~kornia.geometry.camera.pinhole.PinholeCamera`. The two type systems
          are kept separate by design -- this one takes ``Vector`` objects, that one plain tensors -- which is
          recorded in `#4274 <https://github.com/kornia/kornia/issues/4274>`_.

    .. warning::
        :class:`BrownConradyModel`, :class:`KannalaBrandtK3` and :class:`Orthographic` validate their
        parameters and construct, and then every :meth:`project`, :meth:`unproject` and :meth:`matrix` call on
        them raises ``NotImplementedError`` with an empty message, from three independent sites: the
        distortion placeholders ``BrownConradyTransform`` and ``KannalaBrandtK3Transform``, the projection
        placeholder ``OrthographicProjection``, and ``CameraModelBase.matrix``, which those three classes do
        not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_; the behaviour is
        documented as it is.

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
        """Construct CameraModelBase class.

        Args:
            distortion: Distortion type
            projection: Projection type
            image_size: Image size
            params: Camera parameters of shape :math:`(B, N)` or :math:`(N,)`, with
                    :math:`N` fixed by ``distortion``: 4 for
                    :class:`AffineTransform` (pinhole and orthographic), 12 for
                    :class:`BrownConradyTransform`, 8 for
                    :class:`KannalaBrandtK3Transform`.

        Raises:
            ValueError: if ``params`` does not have that shape.

        """
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
        """Return the camera matrix.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.
        """
        raise NotImplementedError

    def K(self) -> torch.Tensor:
        """Return the camera matrix.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.
        """
        return self.matrix()

    def project(self, points: Vector3) -> Vector2:
        """Projects 3D points to 2D camera plane.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        Args:
            points: Vector3 representing 3D points.

        Returns:
            Vector2 representing the projected 2D points.

        Example:
            >>> points = Vector3(torch.Tensor([1.0, 1.0, 1.0]))
            >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
            >>> cam.project(points)
            x: 648.0
            y: 568.0

        """
        return self.distortion.distort(self.params, self.projection.project(points))

    def unproject(self, points: Vector2, depth: torch.Tensor) -> Vector3:
        """Unprojects 2D points from camera plane to 3D.

        See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

        Args:
            points: Vector2 representing 2D points.
            depth: Depth of the points.

        Returns:
            Vector3 representing the unprojected 3D points.

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
    r"""Class to represent Pinhole Camera Model.

    See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

    The pinhole camera model describes the mathematical relationship between
    the coordinates of a point in three-dimensional space and its projection
    onto the image plane of an ideal pinhole camera,
    where the camera aperture is described as a point and no lenses are used to focus light.
    See more: https://en.wikipedia.org/wiki/Pinhole_camera_model

    Example:
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
        >>> cam
        CameraModel(ImageSize(height=480, width=640), PinholeModel, tensor([328., 328., 320., 240.]))

    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        """Construct PinholeModel class.

        Args:
            image_size: Image size
            params: Camera parameters of shape :math:`(B, 4)` of the form :math:`(fx, fy, cx, cy)`.

        """
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
        """Scales the camera model by a scale factor.

        Convention:
            - returns a **new** model whose focal lengths, principal point and image size are multiplied by
              ``scale_factor``: ``fx' = s * fx`` and ``cx' = s * cx``, the half-pixel rule, which is the rule
              :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.scale` applies as well.

        .. warning::
            ``cx' = s * cx`` disagrees with the integer pixel centres the rest of the library enumerates, and
            a tensor ``scale_factor`` rebuilds ``image_size`` with 0-dim tensors in the promoted dtype
            (floating for a floating-point factor) where the constructor took python integers -- a python
            ``int`` keeps ``int`` fields and a python ``float`` gives ``float`` ones. Both are tracked in
            `#4263 <https://github.com/kornia/kornia/issues/4263>`_, the second in its comment thread; they
            are documented as they are.

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
    """Brown Conrady Camera Model.

    .. warning::
        Constructing this model succeeds; :meth:`~kornia.sensors.camera.CameraModelBase.project` and
        :meth:`~kornia.sensors.camera.CameraModelBase.unproject` then raise ``NotImplementedError`` with an
        empty message inside ``BrownConradyTransform``, and
        :meth:`~kornia.sensors.camera.CameraModelBase.matrix` inside ``CameraModelBase.matrix``, which this
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_.
    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        """Construct BrownConradyModel class.

        Args:
            image_size: Image size
            params: Camera parameters of shape :math:`(B, 12)` of the form :math:`(fx, fy, cx, cy, kb0, kb1, kb2, kb3,
                    k1, k2, k3, k4)`.

        """
        if params.shape[-1] != 12 or len(params.shape) > 2:
            raise ValueError("params must be of shape (B, 12) for BROWN_CONRADY Camera")
        super().__init__(BrownConradyTransform(), Z1Projection(), image_size, params)


class KannalaBrandtK3(CameraModelBase):
    """Kannala Brandt K3 Camera Model.

    .. warning::
        Constructing this model succeeds; :meth:`~kornia.sensors.camera.CameraModelBase.project` and
        :meth:`~kornia.sensors.camera.CameraModelBase.unproject` then raise ``NotImplementedError`` with an
        empty message inside ``KannalaBrandtK3Transform``, and
        :meth:`~kornia.sensors.camera.CameraModelBase.matrix` inside ``CameraModelBase.matrix``, which this
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_.
    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        """Construct KannalaBrandtK3 class.

        Args:
            image_size: Image size
            params: Camera parameters of shape :math:`(B, 8)` of the form :math:`(fx, fy, cx, cy, kb0, kb1, kb2, kb3)`.

        """
        if params.shape[-1] != 8 or len(params.shape) > 2:
            raise ValueError("params must be of shape B, 8 for KANNALA_BRANDT_K3 Camera")
        super().__init__(KannalaBrandtK3Transform(), Z1Projection(), image_size, params)


class Orthographic(CameraModelBase):
    """Orthographic Camera Model.

    .. warning::
        Constructing this model succeeds; :meth:`~kornia.sensors.camera.CameraModelBase.project` and
        :meth:`~kornia.sensors.camera.CameraModelBase.unproject` then raise ``NotImplementedError`` with an
        empty message inside ``OrthographicProjection``, and
        :meth:`~kornia.sensors.camera.CameraModelBase.matrix` inside ``CameraModelBase.matrix``, which this
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_.
    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        """Construct Orthographic class.

        Args:
            image_size: Image size
            params: Camera parameters of shape :math:`(B, 4)` of the form :math:`(fx, fy, cx, cy)`.

        """
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError("params must be of shape B, 4 for ORTHOGRAPHIC Camera")
        super().__init__(AffineTransform(), OrthographicProjection(), image_size, params)


CameraModelVariants = Union[PinholeModel, BrownConradyModel, KannalaBrandtK3, Orthographic]


class CameraModel:
    r"""Class to represent camera models.

    See the Convention block on :class:`~kornia.sensors.camera.CameraModelBase`.

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
        """Construct CameraModel class.

        Args:
            image_size: Image size
            model_type: Camera model type
            params: Camera parameters of shape :math:`(B, N)`.

        """
        self._model = get_model_from_type(model_type, image_size, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"CameraModel({self.image_size}, {self._model.__class__.__name__}, {self.params})"
