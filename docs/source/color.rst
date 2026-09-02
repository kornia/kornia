kornia.color
============

.. meta::
   :description: The Color module in Kornia provides a variety of functions for color space conversions, including RGB, HLS, HSV, Lab, and more. It also offers utilities for color maps and Bayer RAW processing.

.. currentmodule:: kornia.color

Color space conversions on float image tensors of shape :math:`(*, C, H, W)` with values in :math:`[0, 1]`,
plus color maps and sepia. Every operation exists as a function and as an ``nn.Module``.

.. note::
   Check a tutorial for color space conversions `here <https://kornia.github.io/tutorials/nbs/hello_world_tutorial.html>`__.

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
