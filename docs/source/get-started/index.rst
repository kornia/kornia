Learn
=====

.. meta::
   :description: Learn Kornia: install the library, follow a quickstart application guide, learn the conventions every function follows, and see what makes the library different.

New to Kornia? Three steps take you from zero to a working pipeline:

.. grid:: 1
   :gutter: 2
   :class-container: kornia-steps

   .. grid-item-card:: 1 · Install
      :link: installation
      :link-type: doc

      ``pip install kornia`` — plus conda and from-source options, and how to check it worked.

   .. grid-item-card:: 2 · QuickStart
      :link: ../applications/image_augmentations
      :link-type: doc

      Pick the application closest to your task — augmentation, matching, stitching, registration,
      detection, prompting or denoising — and adapt its pipeline.

   .. grid-item-card:: 3 · Learn the conventions
      :link: conventions
      :link-type: doc

      ``(B, C, H, W)`` floats in ``[0, 1]``, ``(x, y)`` points vs ``(h, w)`` sizes, degrees vs radians.
      Nearly every subtle Kornia bug is a convention mismatch — read this before writing code.

Looking for a specific function instead? The :doc:`API reference </api>` documents every module, and
:doc:`What is Kornia? <introduction>` gives the one-page overview of the whole library.

.. toctree::
   :caption: Getting started
   :hidden:

   installation
   introduction
   conventions

.. toctree::
   :caption: Why Kornia?
   :hidden:

   gpu-acceleration
   differentiability
   onnx
   edge
   multi-framework-support
   precision
   Performance on Mac chips <performance>

.. toctree::
   :caption: Applications
   :hidden:

   ../applications/image_augmentations
   ../applications/image_matching
   ../applications/image_stitching
   ../applications/image_registration
   ../applications/face_detection
   ../applications/visual_prompting
   ../applications/image_denoising

.. toctree::
   :caption: Resources
   :hidden:

   Tutorials <https://kornia.github.io/tutorials/>
   ONNX, torch.compile & torch.export support <export-support>
