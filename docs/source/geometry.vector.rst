kornia.geometry.vector
======================

.. meta::
   :description: The kornia.geometry.vector module provides the Vector3, Vector2 and Scalar wrappers that kornia's plane, Lie-group and camera-model classes take and return for points, normals and offsets.

.. currentmodule:: kornia.geometry.vector

Tensor wrappers for 3D vectors ``(*, 3)``, 2D vectors ``(*, 2)`` and scalars, which
:class:`~kornia.geometry.plane.Hyperplane`, the Lie groups of ``kornia.geometry.liegroup`` and the camera models of
``kornia.sensors.camera`` take and return. :class:`Vector3` and :class:`Vector2` add coordinate access, dot products
and normalisation, and :class:`Scalar` holds one value per vector. A wrapper does not copy its tensor and is not a
``torch.Tensor``: torch functions and operators accept it, and kornia's tensor functions take ``v.data``.

.. autoclass:: Vector3
   :members:

.. autoclass:: Vector2
   :members:

.. autoclass:: Scalar
   :members:
