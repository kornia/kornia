.. meta::
   :description: Kornia is the open-source geometric computer vision library for Spatial AI and robotics, built on PyTorch: batched, GPU-ready and differentiable image transforms, filters, camera geometry, data augmentation and curated models for detection, segmentation and image matching.

.. raw:: html

   <style>
     /* Landing page only: no "On this page" rail, no header toggle for it. */
     #pst-secondary-sidebar, dialog#pst-secondary-sidebar-modal,
     button.sidebar-toggle.secondary-toggle { display: none !important; }
     /* Hairline separator above each section, matching the link board's line. */
     .bd-article section,
     .kornia-sponsor-notice,
     .kornia-gallery {
       border-top: 1px solid var(--pst-color-border);
       margin-top: 3rem;
       padding-top: 1.75rem;
     }
   </style>

.. container:: kornia-hero

   .. raw:: html

      <h1 class="kornia-hero__title">Computer vision for<br>
      <span class="kornia-accent">robotics &amp; Spatial AI.</span></h1>
      <p class="kornia-hero__tagline">The geometric computer vision library for PyTorch —
      batched, differentiable, GPU-accelerated, with curated pretrained models.</p>

   .. container:: kornia-hero-actions

      .. button-link:: get-started/installation.html
         :color: primary

         Get started

      .. button-link:: api.html
         :color: secondary
         :outline:

         API reference

      .. button-link:: https://github.com/kornia/kornia
         :color: secondary
         :outline:

         :octicon:`mark-github` GitHub

      .. button-link:: https://github.com/kornia/kornia-rs
         :color: secondary
         :outline:

         :octicon:`cpu` Get kornia-rs for edge computing

Why Kornia?
-----------

.. tab-set::
   :class: kornia-why-tabs

   .. tab-item:: :octicon:`zap` GPU-accelerated

      Batched operators run on whatever device the tensor lives on — no flags, no per-image loops.

      .. code-block:: python
         :emphasize-lines: 1

         images = torch.rand(64, 3, 224, 224, device="cuda")  # the whole batch on the GPU
         blurred = kornia.filters.gaussian_blur2d(images, (5, 5), (1.5, 1.5))
         edges = kornia.filters.sobel(blurred)

      .. rst-class:: kornia-proof

      **1,124 → 3,153 images/s** — the same ``RandomGaussianBlur`` at batch 32 on the same Apple M1,
      CPU vs GPU, from the committed :doc:`benchmark runs <get-started/performance>`.

      :doc:`See details <get-started/gpu-acceleration>` :octicon:`arrow-right`

   .. tab-item:: :octicon:`git-compare` Differentiable

      Gradients flow through every operator, so vision ops can live inside your model or your loss.

      .. code-block:: python
         :emphasize-lines: 5

         prediction = torch.rand(2, 3, 64, 64, requires_grad=True)
         target = torch.rand(2, 3, 64, 64)

         loss = kornia.losses.ssim_loss(prediction, target, window_size=5)
         loss.backward()  # gradients flow through the SSIM graph

      .. figure:: _static/img/registration.gif
         :width: 360
         :alt: Two images progressively aligned by gradient descent

         Image registration by pure gradient descent — no labels, no training data.

      :doc:`See details <get-started/differentiability>` :octicon:`arrow-right`

   .. tab-item:: :octicon:`package` Production-ready

      Port any kornia module to ONNX, chain it with pre-exported operators and models, and run the
      combined graph wherever ONNX Runtime runs — no Python, no PyTorch.

      .. code-block:: python
         :emphasize-lines: 1,8

         torch.onnx.export(  # port any kornia module to ONNX...
             kornia.color.RgbToGrayscale(), torch.rand(1, 3, 256, 512), "gray.onnx",
             input_names=["input"], output_names=["output"],
             opset_version=17, dynamo=False,
         )

         pipeline = ONNXSequential(  # ...and chain it with ops from the Hugging Face Hub
             "gray.onnx",
             "hf://operators/kornia.geometry.transform.affwarp.Resize_512x512",
         )
         outputs = pipeline(np.random.randn(1, 3, 256, 512).astype(np.float32))
         pipeline.export("combined.onnx")  # one deployable file

      .. rst-class:: kornia-proof

      **0.015 s vs 0.177 s** per inference — the same exported RT-DETR pipeline on CUDA vs CPU,
      switched with a single ``as_cuda()`` call.

      :doc:`See details <get-started/onnx>` :octicon:`arrow-right`

   .. tab-item:: :octicon:`share-android` Multi-framework

      One API, four frameworks: PyTorch natively, plus TensorFlow, JAX and NumPy through Ivy.

      .. code-block:: python
         :emphasize-lines: 1

         tf_kornia = kornia.to_tensorflow()  # also: kornia.to_jax(), kornia.to_numpy()

         rgb = tf.random.normal((1, 3, 224, 224))
         gray = tf_kornia.color.rgb_to_grayscale(rgb)

      .. rst-class:: kornia-proof

      The first call transpiles and caches each function; later calls run at approximately
      the speed of the native kornia op.

      :doc:`See details <get-started/multi-framework-support>` :octicon:`arrow-right`

.. raw:: html

   <p class="kornia-sponsor-notice">Considering sponsoring? —
   <a href="community/sponsor.html">Inquire now</a></p>

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: kornia-cards kornia-gallery

   .. grid-item-card:: Warp perspective
      :img-top: _static/img/warp_affine.png
      :link: geometry.transform
      :link-type: doc

      ``kornia.geometry`` — rotate, warp and register images with full transform matrices.

   .. grid-item-card:: Augment everything
      :img-top: _static/img/AugmentationSequential.png
      :link: augmentation
      :link-type: doc

      ``kornia.augmentation`` — one sampled transform, applied to image, mask, boxes and keypoints alike.

   .. grid-item-card:: Detect edges
      :img-top: _static/img/canny.png
      :link: filters.edge_detection
      :link-type: doc

      ``kornia.filters`` — Canny, Sobel and Laplacian, differentiable and batched.

   .. grid-item-card:: Match images
      :img-top: _static/img/DISK.png
      :link: feature
      :link-type: doc

      ``kornia.feature`` — DISK, ALIKED, LoFTR and LightGlue find and match keypoints.

   .. grid-item-card:: Recolor and map
      :img-top: _static/img/apply_colormap.png
      :link: color
      :link-type: doc

      ``kornia.color`` — RGB, HSV, Lab, YUV and friends, plus color maps for heat and depth.

   .. grid-item-card:: Enhance and equalize
      :img-top: _static/img/equalize_clahe.png
      :link: enhance
      :link-type: doc

      ``kornia.enhance`` — CLAHE, histogram equalization, gamma and contrast, all trainable.

The rest lives in the :doc:`API reference <api>` — losses and metrics, morphology, camera geometry, image I/O —
and :doc:`Conventions & pitfalls <get-started/conventions>` is the one page to read before writing code.

Ready-to-use models
-------------------

Pretrained models for detection, segmentation and matching — ready in one line:

.. code-block:: python

   import kornia
   from kornia.contrib import RTDETRDetectorBuilder

   image = kornia.io.get_sample_images()[0][None]
   detector = RTDETRDetectorBuilder.build()
   detections = detector(image)  # list of (D, 6) tensors: class id, score, x, y, w, h

.. button-link:: models/index.html
   :color: primary
   :outline:

   Browse the model zoo :octicon:`arrow-right`

.. raw:: html

   <div class="kornia-linkboard">
     <div>
       <p class="kornia-linkboard__title">Docs</p>
       <a href="applications/image_augmentations.html">Quick Start</a>
       <a href="https://kornia.github.io/tutorials/">Tutorial</a>
       <a href="api.html">API</a>
     </div>
     <div>
       <p class="kornia-linkboard__title">Support</p>
       <a href="community/sponsor.html">Sponsor</a>
     </div>
     <div>
       <p class="kornia-linkboard__title">About</p>
       <a href="get-started/governance.html">Team</a>
       <a href="community/community.html">Community Guide</a>
       <a href="https://github.com/kornia/kornia/blob/main/CODE_OF_CONDUCT.md">Conduct</a>
     </div>
     <div>
       <p class="kornia-linkboard__title">Official Links</p>
       <a href="https://twitter.com/kornia_foss">Twitter</a>
       <a href="https://www.linkedin.com/company/kornia/">LinkedIn</a>
       <span class="kornia-linkboard__soon">Newsletter <em>(under construction)</em></span>
     </div>
     <div>
       <p class="kornia-linkboard__title">Official libs</p>
       <a href="https://github.com/kornia/kornia">kornia</a>
       <a href="https://github.com/kornia/kornia-rs">kornia-rs</a>
     </div>
   </div>

.. toctree::
   :hidden:

   Learn <get-started/index>
   API <api>
   Models <models/index>
   About <about>
   Support <community/index>
