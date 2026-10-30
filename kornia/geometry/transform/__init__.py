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

"""Warp, resize, crop, flip and register images and volumes, with their matrices and pyramids.

Images are channels first, ``(B, C, H, W)``, and volumes ``(B, C, D, H, W)``. :func:`warp_perspective`,
:func:`warp_affine` and their 3D forms, the transforms by parameters and the resampling crops take or build matrices
that map **source** pixels to **destination** pixels, with pixel centres at integers and ``x`` the column. The
homography warps (:func:`homography_warp`, :func:`homography_warp3d`, :class:`HomographyWarper`) and the image
registrator default to a destination→source homography in normalized :math:`[-1, 1]` coordinates instead, and the
grid warps (:func:`warp_grid`, :func:`warp_grid3d`) take a destination→source one in the coordinates of the grid.
``align_corners`` defaults differ between functions; see :doc:`Conventions & Pitfalls </get-started/conventions>`.
Every name below is exported here; the submodules in parentheses group them.

Which function to use:

- Warp by a matrix to an output size (``imgwarp``): :func:`warp_perspective`, :func:`warp_affine`, their 3D forms
  :func:`warp_perspective3d` and :func:`warp_affine3d`; by a normalized destination→source homography,
  :func:`homography_warp` (``imgwarp``) and its module form :class:`HomographyWarper` (``homography_warper``); at
  per-pixel coordinates, :func:`remap` (``imgwarp``).
- Build the matrices (``imgwarp``): :func:`get_perspective_transform` from four point pairs,
  :func:`get_rotation_matrix2d`, :func:`get_affine_matrix2d` and their 3D counterparts,
  :func:`get_projective_transform` and :func:`get_affine_matrix3d`, :func:`invert_affine_transform`.
- Rotate, shift, zoom, shear or warp by a matrix at the image size (``affwarp``): :func:`rotate`, :func:`translate`,
  :func:`scale`, :func:`shear`, :func:`affine`, their module forms, and :func:`rotate3d`, :func:`affine3d` for volumes.
- Change the size (``affwarp``): :func:`resize`, :func:`rescale`, :func:`resize_to_be_divisible`.
- Crop (``crop2d``, ``crop3d``): :func:`crop_by_indices` by integer slicing, :func:`crop_and_resize` and
  :func:`crop_by_boxes` by boxes, :func:`crop_by_transform_mat` by a matrix, :func:`center_crop` and
  :class:`CenterCrop2D`, and :func:`crop_and_resize3d` and the other ``3d`` crops for volumes.
- Flip (``flips``): :func:`hflip`, :func:`vflip`, :func:`rot180` and their module forms.
- Pyramids (``pyramid``): :func:`pyrdown`, :func:`pyrup`, :func:`build_pyramid`, :func:`build_laplacian_pyramid`, and
  :class:`ScalePyramid` for scale-space detection.
- Thin-plate splines (``thin_plate_spline``): :func:`get_tps_transform`, :func:`warp_points_tps`,
  :func:`warp_image_tps`.
- Elastic deformation by a smoothed displacement field (``elastic_transform``): :func:`elastic_transform2d`.
- Register two images by optimizing a homography or a similarity (``image_registrator``):
  :class:`~kornia.geometry.transform.image_registrator.ImageRegistrator`.
"""

from .affwarp import *
from .crop2d import *
from .crop3d import *
from .elastic_transform import *
from .flips import *
from .homography_warper import *
from .image_registrator import *
from .imgwarp import *
from .pyramid import *
from .thin_plate_spline import *
