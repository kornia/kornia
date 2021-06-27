kornia.geometry.transform
=========================

.. currentmodule:: kornia.geometry.transform

The functions in this section perform various geometrical transformations of 2D images.

Warp operators
--------------

.. autofunction:: warp_perspective
.. autofunction:: warp_perspective3d
.. autofunction:: warp_affine
.. autofunction:: warp_affine3d
.. autofunction:: warp_image_tps
.. autofunction:: warp_points_tps
.. autofunction:: remap

Image 2d transforms
-------------------

.. autofunction:: affine
.. autofunction:: rotate
.. autofunction:: translate
.. autofunction:: scale
.. autofunction:: shear
.. autofunction:: hflip
.. autofunction:: vflip
.. autofunction:: rot180
.. autofunction:: resize
.. autofunction:: rescale
.. autofunction:: elastic_transform2d
.. autofunction:: pyrdown
.. autofunction:: pyrup
.. autofunction:: build_pyramid

Matrix transformations
----------------------

.. autofunction:: get_perspective_transform
.. autofunction:: get_perspective_transform3d
.. autofunction:: get_projective_transform
.. autofunction:: get_rotation_matrix2d
.. autofunction:: get_shear_matrix2d
.. autofunction:: get_shear_matrix3d
.. autofunction:: get_affine_matrix2d
.. autofunction:: get_affine_matrix3d
.. autofunction:: invert_affine_transform
.. autofunction:: projection_from_Rt
.. autofunction:: get_tps_transform
.. autofunction:: normalize_homography
.. autofunction:: denormalize_homography

Crop operators
--------------

.. autofunction:: crop_by_boxes
.. autofunction:: center_crop
.. autofunction:: crop_and_resize

Bounding Box
------------

.. autofunction:: bbox_to_mask
.. autofunction:: infer_box_shape
.. autofunction:: validate_bboxes

Module
------

.. autoclass:: Rotate
.. autoclass:: Translate
.. autoclass:: Scale
.. autoclass:: Shear
.. autoclass:: PyrDown
.. autoclass:: PyrUp
.. autoclass:: ScalePyramid
.. autoclass:: Hflip
.. autoclass:: Vflip
.. autoclass:: Rot180
.. autoclass:: Resize
.. autoclass:: Rescale
.. autoclass:: Affine
