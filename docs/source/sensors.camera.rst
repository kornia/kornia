kornia.sensors.camera
======================

.. meta::
   :description: The `kornia.sensors.camera` module provides differentiable Pinhole, Brown-Conrady, Kannala-Brandt K3, and Orthographic camera models, together with reusable distortion and projection components for defining custom camera models.

.. currentmodule:: kornia.sensors.camera

.. warning::
   :mod:`kornia.sensors.camera` is an experimental API and is subject to change. Once finished, it will subsume :mod:`kornia.geometry.camera`, although today the two are kept separate by design -- they share the mathematical Pinhole mapping but differ in input types, camera-axis broadcasting, projection rounding, and zero/near-zero depth handling (see the Convention block on :class:`CameraModelBase`), which is recorded in `#4274 <https://github.com/kornia/kornia/issues/4274>`_.

The objective of :mod:`kornia.sensors.camera` is to express well-known camera models such as Pinhole, Kannala Brandt, and others in terms of distortion and projection types while ensuring differentiability.
We also aim to equip the user with tools to define custom camera models.

Defining a `Pinhole` camera model is as simple as:

.. code:: python

    import torch
    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModel, CameraModelType

    params = torch.tensor([328., 328., 320., 240.])  # fx, fy, cx, cy
    cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, params)

To define a custom camera model based on distortion and projection types, one can use the :class:`CameraModelBase` API:

.. code:: python

    import torch
    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModelBase
    from kornia.sensors.camera.distortion_model import AffineTransform
    from kornia.sensors.camera.projection_model import Z1Projection

    params = torch.tensor([328., 328., 320., 240.])
    cam = CameraModelBase(AffineTransform(), Z1Projection(), ImageSize(480, 640), params)

.. note::
   The built-in camera models are Pinhole, Brown-Conrady, Kannala-Brandt K3, and Orthographic.

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
