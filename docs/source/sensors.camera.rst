kornia.sensors.camera
======================

.. meta::
   :description: The `kornia.sensors.camera` module provides tools to define and manipulate various camera models, including the Pinhole model. It allows users to specify distortion and projection types in a differentiable way. Pinhole is the only model that works today; the Brown-Conrady, Kannala-Brandt and orthographic models are exported but not yet implemented. It also enables users to define custom camera models using distortion and projection types.

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
   Pinhole is the model that works today. :class:`BrownConradyModel`, :class:`KannalaBrandtK3` and :class:`Orthographic` validate their parameters and construct, and then every ``project``, ``unproject`` and ``matrix`` call on them raises ``NotImplementedError``; implementing them is tracked in `#4284 <https://github.com/kornia/kornia/issues/4284>`_.

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


Projections
-----------

.. currentmodule:: kornia.sensors.camera.projection_model

.. autoclass:: Z1Projection
    :members:
