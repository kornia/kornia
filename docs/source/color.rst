kornia.color
============

.. meta::
   :description: The Color module in Kornia provides a variety of functions for color space conversions, including RGB, HLS, HSV, Lab, and more. It also offers utilities for color maps and Bayer RAW processing.

.. currentmodule:: kornia.color

Color space conversions on tensors with channels at axis -3, plus color maps and sepia.
Each operation documents its channel order and range.

Conventions
-----------

Color tensors are channel-first, :math:`(*, C, H, W)`. Float RGB inputs are unit-range nonlinear sRGB,
except that :func:`rgb_to_xyz` takes linear RGB and the ``*rgb255*`` helpers use :math:`[0, 255]`.
:func:`rgb_to_linear_rgb` and :func:`linear_rgb_to_rgb` apply the sRGB transfer curve. HLS and HSV
hue is measured in radians. The conversion pages define the remaining channel orders and ranges.

.. note::
   Check a tutorial for color space conversions `here <https://www.kornia.org/tutorials/nbs/hello_world_tutorial.html>`__.

.. list-table::
   :widths: 30 70

   * - :doc:`Color conversion <color.conversions>`
     - Conversions between :doc:`grayscale <color.grayscale>`, :doc:`RGB <color.rgb>`, :doc:`BGR <color.bgr>`,
       :doc:`RGBA <color.rgba>`, :doc:`linear RGB <color.linear_rgb>`, :doc:`HLS <color.hls>`,
       :doc:`HSV <color.hsv>`, :doc:`Lab <color.lab>`, :doc:`Luv <color.luv>`, :doc:`XYZ <color.xyz>`,
       :doc:`YCbCr <color.ycbcr>`, :doc:`YUV <color.yuv>` and :doc:`Bayer RAW <color.raw>`.
   * - :doc:`Colormap <color.colormap>`
     - Render single-channel images (depth, heat, edges) with a color map.
   * - :doc:`Sepia <color.sepia>`
     - The sepia tone effect.

.. toctree::
   :hidden:

   color conversion <color.conversions>
   colormap <color.colormap>
   sepia <color.sepia>
