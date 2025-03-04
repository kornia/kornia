kornia.geometry.epipolar
========================

.. meta::
   :name: description
   :content: "The kornia.geometry.epipolar module provides essential tools for working with epipolar geometry, crucial in tasks like Structure from Motion (SfM). It includes functions for computing the essential and fundamental matrices, decomposing them, and deriving relative camera motion. The module also offers various metrics for evaluating epipolar constraints, triangulating 3D points, and handling projection transformations. Additionally, it supports functions for normalizing points and transformations, calculating epipolar distances, and generating camera intrinsics."

.. currentmodule:: kornia.geometry.epipolar

Module with useful functionalities for epipolar geometry used by Structure from Motion

.. image:: data/epipolar_geometry.svg.png


Essential
---------
.. autofunction:: find_essential
.. autofunction:: essential_from_fundamental
.. autofunction:: essential_from_Rt
.. autofunction:: decompose_essential_matrix
.. autofunction:: motion_from_essential
.. autofunction:: motion_from_essential_choose_solution
.. autofunction:: relative_camera_motion

Fundamental
-----------

.. autofunction:: find_fundamental
.. autofunction:: fundamental_from_essential
.. autofunction:: fundamental_from_projections
.. autofunction:: compute_correspond_epilines
.. autofunction:: normalize_points
.. autofunction:: normalize_transformation
.. autofunction:: get_perpendicular
.. autofunction:: get_closest_point_on_epipolar_line


Metrics
-------

.. autofunction:: sampson_epipolar_distance
.. autofunction:: symmetrical_epipolar_distance
.. autofunction:: left_to_right_epipolar_distance
.. autofunction:: right_to_left_epipolar_distance

Projection
----------

.. autofunction:: projection_from_KRt
.. autofunction:: projections_from_fundamental
.. autofunction:: intrinsics_like
.. autofunction:: scale_intrinsics
.. autofunction:: random_intrinsics

Numeric
-------

.. autofunction:: cross_product_matrix


Triangulation
-------------

.. autofunction:: triangulate_points
