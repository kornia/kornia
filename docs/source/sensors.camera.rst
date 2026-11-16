kornia.sensors.camera
======================

.. meta::
   :description: The `kornia.sensors.camera` module provides differentiable Pinhole, Brown-Conrady, Kannala-Brandt K3, Orthographic and Double Sphere camera models, together with reusable distortion and projection components for defining custom camera models.

.. currentmodule:: kornia.sensors.camera

.. warning::
   :mod:`kornia.sensors.camera` is an experimental API and is subject to change. Once finished, it will subsume :mod:`kornia.geometry.camera`, although the two are kept separate by design -- they share the mathematical Pinhole mapping but differ in input types, camera-axis broadcasting and zero-depth handling (see the Convention block and warning on :class:`CameraModelBase`), which is recorded in `#4274 <https://github.com/kornia/kornia/issues/4274>`_.

:mod:`kornia.sensors.camera` expresses a camera model as a projection, from camera-frame points to the image plane,
followed by a distortion, from the image plane to pixels. Points are
:class:`~kornia.geometry.vector.Vector3` and pixels :class:`~kornia.geometry.vector.Vector2`, and ``params`` is a flat
``(N,)`` or batched ``(B, N)`` vector that starts with ``[fx, fy, cx, cy]``.

The Double Sphere model is a complete 3D projection, not a distortion applied after projection to ``z = 1``. It can
therefore represent valid rays with negative camera-frame ``z``. :class:`DoubleSphereModel.unproject_rays` returns
unit-length bearings, and :meth:`DoubleSphereModel.unproject` takes Euclidean distance along each bearing. See the
`Double Sphere paper <https://arxiv.org/abs/1807.08957>`_ for its equations and valid domains.

- Build a built-in model by name with :class:`CameraModel` and a :class:`CameraModelType`, or with its class:
  :class:`PinholeModel` (no lens distortion), :class:`BrownConradyModel` (OpenCV's radial and tangential
  distortion), :class:`KannalaBrandtK3` (fisheye), :class:`Orthographic` (parallel projection) or
  :class:`DoubleSphereModel` (full 3D wide-angle projection).
- Compose a custom model with :class:`CameraModelBase` from one of the distortions and one of the projections below.
- For cameras with extrinsics, on plain tensors, use :class:`~kornia.geometry.camera.pinhole.PinholeCamera`.

Defining a pinhole camera model:

.. code:: python

    import torch
    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModel, CameraModelType

    params = torch.tensor([328., 328., 320., 240.])  # fx, fy, cx, cy
    cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, params)

Defining the same model from its distortion and projection with :class:`CameraModelBase`:

.. code:: python

    import torch
    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModelBase
    from kornia.sensors.camera.distortion_model import AffineTransform
    from kornia.sensors.camera.projection_model import Z1Projection

    params = torch.tensor([328., 328., 320., 240.])
    cam = CameraModelBase(AffineTransform(), Z1Projection(), ImageSize(480, 640), params)

Defining a Double Sphere camera model and recovering unit bearings:

.. code:: python

    import torch
    from kornia.geometry.vector import Vector2, Vector3
    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModel, CameraModelType

    params = torch.tensor([300., 320., 160., 120., 0.5, 0.4])  # fx, fy, cx, cy, xi, alpha
    cam = CameraModel(ImageSize(240, 320), CameraModelType.DOUBLE_SPHERE, params)
    pixels = cam.project(Vector3(torch.tensor([0.3, -0.2, 2.])))
    bearings = cam.unproject_rays(pixels)

``DoubleSphereModel.unproject`` scales these bearings using Euclidean distance along each ray; this is distinct from
the camera-frame ``z`` depth used by :class:`CameraModelBase`.

.. autoclass:: CameraModelBase
    :members:

.. autoclass:: CameraModel
    :members:

.. autoclass:: CameraModelType
    :members:

.. autoclass:: PinholeModel
    :members:

.. autoclass:: BrownConradyModel
    :members:

.. autoclass:: KannalaBrandtK3
    :members:

.. autoclass:: Orthographic
    :members:

.. autoclass:: DoubleSphereModel
    :members:

Distortions
-----------

.. currentmodule:: kornia.sensors.camera.distortion_model

.. autoclass:: AffineTransform
    :members:

.. autoclass:: BrownConradyTransform
    :members:

.. autoclass:: KannalaBrandtK3Transform
    :members:


Projections
-----------

.. currentmodule:: kornia.sensors.camera.projection_model

.. autoclass:: Z1Projection
    :members:

.. autoclass:: OrthographicProjection
    :members:
