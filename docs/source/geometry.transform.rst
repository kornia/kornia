kornia.geometry.transform
=========================

.. meta::
   :description: The kornia.geometry.transform module provides a comprehensive set of functions for performing various geometric transformations on 2D and 3D images. It includes operators for warping, affine, and perspective transformations, as well as tools for resizing, scaling, rotating, and shearing images. Additionally, it supports advanced image registration techniques, pyramid building, and non-rigid transformations such as elastic transforms. This module is essential for tasks requiring image manipulation and alignment in computer vision and image processing workflows.

.. currentmodule:: kornia.geometry.transform

The functions in this section warp, resize, crop, flip and register 2D images and 3D volumes. The warps take a
matrix that maps source pixels to destination pixels (:func:`warp_perspective`, :func:`warp_affine` and their 3D
forms), except :func:`homography_warp` and :func:`homography_warp3d`, which take a destination→source homography in
normalized :math:`[-1, 1]` coordinates by default. The 2D image transforms :func:`rotate`, :func:`translate`,
:func:`scale` and :func:`shear` take parameters instead of a matrix and :func:`affine` takes a matrix, all at the
image size; :func:`resize` and :func:`rescale` change the size. ``align_corners`` defaults differ between these
functions; see :doc:`Conventions & Pitfalls </get-started/conventions>`.

Warp operators
--------------

Warp images ``(B, C, H, W)`` or volumes ``(B, C, D, H, W)`` to the output size ``dsize`` by a source→destination
pixel homography (:func:`warp_perspective`, :func:`warp_perspective3d`) or affine matrix (:func:`warp_affine`,
:func:`warp_affine3d`), or by a destination→source homography in normalized coordinates (:func:`homography_warp`,
:func:`homography_warp3d`). Sample an image at per-pixel source coordinates with :func:`remap`, or warp it by
thin-plate-spline control points with :func:`warp_image_tps` (and points with :func:`warp_points_tps`).
:func:`warp_grid` and :func:`warp_grid3d` return the warped sampling grid without sampling.

.. autofunction:: warp_perspective
.. autofunction:: warp_perspective3d
.. autofunction:: warp_affine
.. autofunction:: warp_affine3d
.. autofunction:: warp_image_tps
.. autofunction:: warp_points_tps
.. autofunction:: warp_grid
.. autofunction:: warp_grid3d
.. autofunction:: remap
.. autofunction:: homography_warp
.. autofunction:: homography_warp3d

Image 2D transforms
-------------------

Rotate, shift, zoom or shear an image by parameters (:func:`rotate`, :func:`translate`, :func:`scale`,
:func:`shear`), or warp it by an affine matrix (:func:`affine`), at the input size; change the size with
:func:`resize`, :func:`rescale` or :func:`resize_to_be_divisible`. Mirror or turn it by reversing axes, without
interpolation (:func:`hflip`, :func:`vflip`, :func:`rot180`); deform it by a smoothed displacement field
(:func:`elastic_transform2d`); blur and resize it by one pyramid level (:func:`pyrdown`, :func:`pyrup`), or build a
Gaussian or Laplacian pyramid (:func:`build_pyramid`, :func:`build_laplacian_pyramid`, with the integer helpers
:func:`is_powerof_two` and :func:`find_next_powerof_two`).

.. autofunction:: affine
.. autofunction:: rotate
.. autofunction:: translate
.. autofunction:: scale
.. autofunction:: shear
.. autofunction:: hflip
.. autofunction:: vflip
.. autofunction:: rot180
.. autofunction:: resize
.. autofunction:: resize_to_be_divisible
.. autofunction:: rescale
.. autofunction:: elastic_transform2d
.. autofunction:: pyrdown
.. autofunction:: pyrup
.. autofunction:: build_pyramid
.. autofunction:: build_laplacian_pyramid
.. autofunction:: is_powerof_two
.. autofunction:: find_next_powerof_two

Image 3D transforms
-------------------

Warp a volume by an affine matrix (:func:`affine3d`) or rotate it about its centre by an axis-angle vector
(:func:`rotate3d`), at the input size.

.. autofunction:: affine3d
.. autofunction:: rotate3d

Matrix transformations
----------------------

Build the matrices the warps take: a homography from point pairs (:func:`get_perspective_transform`, and
:func:`get_perspective_transform3d` from box corners), an affine matrix from a rotation, a translation, a shear or
all of them (:func:`get_rotation_matrix2d`, :func:`get_translation_matrix2d`, :func:`get_shear_matrix2d`,
:func:`get_affine_matrix2d`, and in 3D :func:`get_projective_transform`, :func:`get_shear_matrix3d`,
:func:`get_affine_matrix3d`), the inverse of an affine matrix (:func:`invert_affine_transform`) and ``[R | t]``
(:func:`projection_from_Rt`). :func:`get_tps_transform` fits the thin-plate spline that :func:`warp_image_tps` takes.

.. autofunction:: get_perspective_transform
.. autofunction:: get_perspective_transform3d
.. autofunction:: get_projective_transform
.. autofunction:: get_rotation_matrix2d
.. autofunction:: get_translation_matrix2d
.. autofunction:: get_shear_matrix2d
.. autofunction:: get_shear_matrix3d
.. autofunction:: get_affine_matrix2d
.. autofunction:: get_affine_matrix3d
.. autofunction:: invert_affine_transform
.. autofunction:: projection_from_Rt
.. autofunction:: get_tps_transform

Crop operators
--------------

Crop ``(B, C, H, W)`` images by integer slicing of axis-aligned boxes (:func:`crop_by_indices`), by resampling a
four-corner box to a fixed size (:func:`crop_and_resize`) or onto a second box whose size sets the output
(:func:`crop_by_boxes`), by a source→destination pixel matrix (:func:`crop_by_transform_mat`), or at the centre
(:func:`center_crop`, and its module form :class:`CenterCrop2D`). A box is ``(B, 4, 2)``: its ``(x, y)`` corners
clockwise from the top-left, as inclusive pixel positions.

.. autofunction:: crop_by_indices
.. autofunction:: crop_by_boxes
.. autofunction:: crop_by_transform_mat
.. autofunction:: center_crop
.. autofunction:: crop_and_resize

Crop operators (3D)
--------------------

Crop ``(B, C, D, H, W)`` volumes by ``(B, 8, 3)`` boxes, resampled to a fixed size (:func:`crop_and_resize3d`) or onto
a second box (:func:`crop_by_boxes3d`), by a source→destination voxel matrix (:func:`crop_by_transform_mat3d`), or at
the centre (:func:`center_crop3d`). Both ``align_corners`` settings sample the same voxels inside the volume.

.. autofunction:: crop_by_boxes3d
.. autofunction:: crop_by_transform_mat3d
.. autofunction:: center_crop3d
.. autofunction:: crop_and_resize3d

Modules
-------

The ``torch.nn.Module`` classes. Module forms that store their arguments: the 2D transforms (:class:`Rotate`,
:class:`Translate`, :class:`Scale`, :class:`Shear`, :class:`Affine`), the resizes (:class:`Resize`, :class:`Rescale`),
the pyramid steps (:class:`PyrDown`, :class:`PyrUp`) and the centre crop (:class:`CenterCrop2D`); module forms that
take no arguments: the flips (:class:`Hflip`, :class:`Vflip`, :class:`Rot180`). :class:`ScalePyramid` builds
scale-space octaves, :class:`HomographyWarper` warps by a normalized homography, and :class:`BaseWarper` is the
interface for writing another warper.

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
.. autoclass:: BaseWarper
.. autoclass:: HomographyWarper
.. autoclass:: CenterCrop2D

Image registration
------------------

.. image:: _static/img/registration.gif
  :width: 400
  :alt: Image registration with the ImageRegistrator module

See also the :doc:`image registration application page <applications/image_registration>`.

.. automodule:: kornia.geometry.transform.image_registrator
    :members:
