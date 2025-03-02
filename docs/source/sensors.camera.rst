kornia.sensors.camera
======================

.. meta::
   :name: description
   :content: "The `kornia.sensors.camera` module provides tools to define and manipulate various camera models, including the Pinhole model. It allows users to specify distortion and projection types in a differentiable way. While currently supporting only the Pinhole model, the module aims to extend its support to other models like Kannala Brandt and Orthographic. It also enables users to define custom camera models using distortion and projection types."

.. currentmodule:: kornia.sensors.camera

.. warning::
   :mod:`kornia.sensors.camera` is an experimental API and is subject to change. Once finished, it will subsume :mod:`kornia.geometry.camera`

The objective of :mod:`kornia.sensors.camera` is to express well-known camera models such as Pinhole, Kannala Brandt, and others in terms of distortion and projection types while ensuring differentiability.
We also aim to equip the user with tools to define custom camera models.
As of now, only the Pinhole model is supported.

Defining a `Pinhole` camera model is as simple as:

.. code:: python

    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModel, CameraModelType

    params = torch.tensor([328., 328., 320., 240.]) # fx, fy, cx, cy
    cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, params)

To define a custom camera model based on distortion and projection types, one can use the :class:`CameraModelBase` api:

.. code:: python

    from kornia.image import ImageSize
    from kornia.sensors.camera import CameraModelBase
    from kornia.sensors.camera.distortion_model import AffineTransform
    from kornia.sensors.camera.projection_model import Z1Projection

    params = torch.tensor([328., 328., 320., 240.])
    cam = CameraModelBase(AffineTransform(), Z1Projection(), ImageSize(480, 640), params)


.. note::
   At the moment, the only supported model is Pinhole. However, we plan to add Kannala Brandt, Orthographic, and other models in the near future.

.. autoclass:: CameraModelBase
    :members:

.. autoclass:: CameraModel
    :members:

.. autoclass:: CameraModelType
    :members:

.. autoclass:: PinholeModel
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
