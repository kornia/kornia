kornia.geometry.quaternion
==========================

.. meta::
   :description: The kornia.geometry.quaternion module provides tools for working with quaternions, a mathematical concept widely used in 3D geometry and computer vision. The Quaternion class allows for quaternion manipulation, including conversion between different representations like axis-angle and rotation matrices. This module is essential for operations involving 3D rotations and transformations.

.. currentmodule:: kornia.geometry.quaternion

:class:`Quaternion` holds ``(*, 4)`` quaternions in ``(w, x, y, z)`` order, multiplies them by the Hamilton product and
converts them to and from rotation matrices, rotation vectors and Euler angles; :func:`average_quaternions` returns the
mean rotation of many. A quaternion does not act on points: :class:`~kornia.geometry.liegroup.So3` wraps one to rotate
points and adds ``exp``, ``log`` and Jacobians, and :class:`~kornia.geometry.liegroup.Se3` adds a translation. For the
same conversions on plain tensors, see :doc:`geometry.conversions`.

Module state
------------

Quaternion data given as a plain tensor is a persistent buffer under ``_data``; an ``nn.Parameter`` stays a
parameter under the same key. ``So3`` saves its rotation as ``_q._data`` and ``Se3`` as ``_rotation._q._data``,
prefixed by the enclosing module's attribute names when nested. ``load_state_dict(strict=True)`` raises on a state
dict that lacks these keys; with ``strict=False`` they are reported as missing and the target keeps its own rotation.

.. autoclass:: Quaternion
   :members:
   :special-members:

.. autofunction:: average_quaternions
