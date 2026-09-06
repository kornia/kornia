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
          8 for ``KANNALA_BRANDT_K3``, laid out as each constructor documents. An unbatched ``(N,)`` vector
          and a batched :math:`(B, N)` one are both accepted; another length, or a rank above 2, raises
          ``ValueError``.
        - :meth:`matrix` and its alias :meth:`K` return the :math:`(*, 3, 3)` intrinsics
          ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``, carrying the batch axis of ``params`` -- not the
          :math:`(B, 4, 4)` matrix :class:`~kornia.geometry.camera.pinhole.PinholeCamera` stores.
          :class:`PinholeModel` implements it; ``CameraModelBase.matrix`` itself raises
          ``NotImplementedError``.
        - :meth:`project` is ``distortion.distort(projection.project(points))`` and :meth:`unproject` the
          reverse, ``projection.unproject(distortion.undistort(points), depth)``. ``depth`` is the
          camera-frame ``z``: :class:`~kornia.sensors.camera.projection_model.Z1Projection` multiplies the
          :math:`z = 1` point by it, so the third coordinate of the result is the ``depth`` that was passed
          in, and not a Euclidean ray length.
        - on the Pinhole path the numbers are those of :doc:`kornia.geometry.camera </geometry.camera>`:
          :meth:`project` matches :func:`~kornia.geometry.camera.perspective.project_points` and
          :meth:`unproject` matches :func:`~kornia.geometry.camera.perspective.unproject_points` on the ``K``
          built from the same ``[fx, fy, cx, cy]``, so the pixels are on the integer-centre grid described in
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
        documented as it is and pinned by
        ``test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284`` in
        ``tests/sensors/camera/test_camera_model.py``.

    .. warning::
        :meth:`unproject` forwards ``depth`` to the projection, and
        :class:`~kornia.sensors.camera.projection_model.Z1Projection` promotes a python ``float`` or ``int``
        with ``torch.Tensor([depth])``, which ignores the device and the dtype of ``points``: on an
        accelerator the multiply that follows raises ``RuntimeError``. Pass a tensor built on the device of
        ``points``. Tracked in `#4313 <https://github.com/kornia/kornia/issues/4313>`_ and pinned by
        ``test_wart_unproject_with_a_python_scalar_depth_builds_a_cpu_tensor_4313`` in
        ``tests/sensors/camera/test_projection_model.py``.

    Example:
        >>> params = torch.Tensor([328., 328., 320., 240.])
        >>> cam = CameraModelBase(BrownConradyTransform(), Z1Projection(), ImageSize(480, 640), params)
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
            params: Camera parameters of shape :math:`(B, 4)`
                    for PINHOLE Camera, :math:`(B, 12)`
                    for Brown Conrady, :math:`(B, 8)`
                    for Kannala Brandt K3.

        """
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
            the rebuilt ``image_size`` carries 0-dim floating tensors where the constructor took python
            integers. Both are tracked in `#4263 <https://github.com/kornia/kornia/issues/4263>`_, the second
            in its comment thread; they are documented as they are and pinned by
            ``test_wart_scale_rescales_the_principal_point_by_the_half_pixel_rule_4263`` and
            ``test_wart_scale_turns_the_image_size_fields_into_tensors_4263`` in
            ``tests/sensors/camera/test_camera_model.py``.

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
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_ and
        pinned by ``test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284`` in
        ``tests/sensors/camera/test_camera_model.py``.
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
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_ and
        pinned by ``test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284`` in
        ``tests/sensors/camera/test_camera_model.py``.
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
        class does not override. Tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_ and
        pinned by ``test_wart_the_three_non_pinhole_models_construct_and_then_raise_4284`` in
        ``tests/sensors/camera/test_camera_model.py``.
    """

    def __init__(self, image_size: ImageSize, params: torch.Tensor) -> None:
        """Construct Orthographic class.

        Args:
            image_size: Image size
            params: Camera parameters of shape :math:`(B, 4)` of the form :math:`(fx, fy, cx, cy)`.

        """
        super().__init__(AffineTransform(), OrthographicProjection(), image_size, params)
        if params.shape[-1] != 4 or len(params.shape) > 2:
            raise ValueError("params must be of shape B, 4 for ORTHOGRAPHIC Camera")


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
