Color conversion
================

.. currentmodule:: kornia.color

Conversions between color spaces, on float image tensors of shape :math:`(*, C, H, W)` with values in
:math:`[0, 1]`. Every conversion exists as a function and as an ``nn.Module``; each page below documents
one color scheme.

.. toctree::
   :maxdepth: 1

   grayscale <color.grayscale>
   rgb <color.rgb>
   bgr <color.bgr>
   rgba <color.rgba>
   linear rgb <color.linear_rgb>
   hls <color.hls>
   hsv <color.hsv>
   lab <color.lab>
   luv <color.luv>
   xyz <color.xyz>
   ycbcr <color.ycbcr>
   yuv <color.yuv>
   bayer raw <color.raw>

All functions
-------------

.. autosummary::
   :nosignatures:

   rgb_to_grayscale
   bgr_to_grayscale
   grayscale_to_rgb
   rgb_to_bgr
   bgr_to_rgb
   rgb_to_rgb255
   rgb255_to_rgb
   rgb255_to_normals
   normals_to_rgb255
   rgb_to_rgba
   bgr_to_rgba
   rgba_to_rgb
   rgba_to_bgr
   rgb_to_linear_rgb
   linear_rgb_to_rgb
   rgb_to_hls
   hls_to_rgb
   rgb_to_hsv
   hsv_to_rgb
   rgb_to_lab
   lab_to_rgb
   rgb_to_luv
   luv_to_rgb
   rgb_to_xyz
   xyz_to_rgb
   rgb_to_ycbcr
   ycbcr_to_rgb
   rgb_to_yuv
   yuv_to_rgb
   rgb_to_yuv420
   yuv420_to_rgb
   rgb_to_yuv422
   yuv422_to_rgb
   rgb_to_raw
   raw_to_rgb
   raw_to_rgb_2x2_downscaled
