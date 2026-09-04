.. meta::
   :description: Kornia is the open-source differentiable computer vision library for PyTorch, with GPU-accelerated image processing, geometry, augmentation and vision models.

.. title:: Differentiable Computer Vision for PyTorch

Kornia: Differentiable Computer Vision for PyTorch
==================================================

.. raw:: html

   <style>
     /* Landing page only. conf.py already gives this page no right-rail items, but pydata still
        renders the (empty) rail column and its header toggle; hide both. */
     #pst-secondary-sidebar, dialog#pst-secondary-sidebar-modal,
     button.sidebar-toggle.secondary-toggle { display: none !important; }
     /* Room for the two-column hero; the theme caps articles at 960px. */
     .bd-main .bd-content .bd-article-container { max-width: 1240px; }
     /* The document title provides the semantic H1 and browser title; the hero is its visible presentation. */
     section#kornia-differentiable-computer-vision-for-pytorch > h1:first-child {
       border: 0; clip: rect(0 0 0 0); height: 1px; margin: -1px; overflow: hidden;
       padding: 0; position: absolute; white-space: nowrap; width: 1px;
     }
     /* Hairline separator above each section, matching the link board's line. */
     section#kornia-differentiable-computer-vision-for-pytorch > section {
       border-top: 1px solid var(--pst-color-border);
       margin-top: 3rem;
       padding-top: 1.75rem;
     }
   </style>

.. grid:: 1 1 1 2
   :gutter: 4
   :class-container: kornia-hero
   :padding: 0

   .. grid-item::
      :columns: 12 12 12 5
      :class: kornia-hero__copy

      .. raw:: html

         <div class="kornia-pip" title="Install from PyPI">
           <span class="kornia-pip__prompt" aria-hidden="true">&gt;_</span>
           <code class="kornia-pip__cmd">pip install kornia</code>
           <button type="button" class="kornia-pip__copy" data-copy="pip install kornia"
                   aria-label="Copy install command"><i class="fa-regular fa-copy"></i></button>
         </div>
         <p class="kornia-hero__title">Computer vision for<br>
         <span class="kornia-accent">robotics &amp; Spatial AI.</span></p>

      .. container:: kornia-hero-actions

         .. button-link:: get-started/installation.html
            :color: primary

            Get started

         .. button-link:: api.html
            :color: secondary
            :outline:

            API reference

      .. raw:: html

         <p class="kornia-hero__links">
           <a href="https://github.com/kornia/kornia-rs"><i class="fa-solid fa-microchip"></i> Also available: kornia-rs for edge devices</a>
         </p>

   .. grid-item::
      :columns: 12 12 12 7
      :class: kornia-hero__demo

      .. raw:: html

         <p class="kornia-hero__eyebrow">Why Kornia?</p>

      .. tab-set::
         :class: kornia-why-tabs

         .. tab-item:: :octicon:`zap` GPU-accelerated

            .. Bar chart drawn at build time by docs/generate_benchmarks.py from benchmarks/results/,
               so the throughput figures track the committed run.

            .. raw:: html
               :file: _generated/hero-benchmark.html

            .. code-block:: python

               edges = kornia.filters.sobel(images.cuda())  # whole batch, one call

            **Same code, any device.** Operators run wherever the tensor lives — no flags, no per-image loops.

            :doc:`See details <get-started/gpu-acceleration>` :octicon:`arrow-right`

         .. tab-item:: :octicon:`git-compare` Differentiable

            .. raw:: html

               <div class="kornia-tab-visual kornia-tab-visual--image">
                 <img src="_static/img/registration.gif" alt="Two images progressively aligned by gradient descent">
                 <div class="kornia-tab-visual__aside">
                   <p class="kornia-tab-visual__eyebrow">gradient descent on a homography</p>
                   <code>warped = warp_perspective(src, H, size)</code>
                   <code>loss = (warped - dst).abs().mean()</code>
                   <code>loss.backward(); optimizer.step()</code>
                   <p class="kornia-tab-visual__eyebrow">repeat until aligned</p>
                 </div>
               </div>

            **Registration by pure gradient descent.** Gradients flow through every operator, so vision ops
            can sit inside your model or your loss — no labels, no training data.

            :doc:`See details <get-started/differentiability>` :octicon:`arrow-right`

         .. tab-item:: :octicon:`package` Production-ready

            .. raw:: html

               <div class="kornia-tab-visual">
                 <svg class="kornia-illo" viewBox="0 0 640 220" role="img"
                      aria-label="A kornia module is exported to ONNX, chained with Hub-hosted operators, and runs on ONNX Runtime">
                   <defs>
                     <marker id="kornia-illo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
                       <path d="M0,0 L10,5 L0,10 z" class="illo-arrowhead"/>
                     </marker>
                   </defs>
                   <text x="24" y="36" class="illo-title">PyTorch module → ONNX graph → one deployable file</text>
                   <rect x="24" y="66" width="160" height="72" rx="10" class="illo-box"/>
                   <text x="104" y="96" text-anchor="middle" class="illo-value">kornia module</text>
                   <text x="104" y="120" text-anchor="middle" class="illo-mono">RgbToGrayscale()</text>
                   <path d="M188,102 L226,102" class="illo-line" marker-end="url(#kornia-illo-arrow)"/>
                   <rect x="232" y="66" width="140" height="72" rx="10" class="illo-box"/>
                   <text x="302" y="96" text-anchor="middle" class="illo-value">torch.onnx</text>
                   <text x="302" y="120" text-anchor="middle" class="illo-mono">gray.onnx</text>
                   <path d="M376,102 L414,102" class="illo-line" marker-end="url(#kornia-illo-arrow)"/>
                   <rect x="420" y="66" width="196" height="72" rx="10" class="illo-primary"/>
                   <text x="518" y="96" text-anchor="middle" class="illo-value illo-on-primary">ONNXSequential</text>
                   <text x="518" y="120" text-anchor="middle" class="illo-mono illo-on-primary">+ hf://…/Resize_512x512</text>
                   <text x="24" y="182" class="illo-label">runs on</text>
                   <rect x="90" y="164" width="120" height="28" rx="14" class="illo-pill"/>
                   <text x="150" y="183" text-anchor="middle" class="illo-pill-text">ONNX Runtime</text>
                   <rect x="220" y="164" width="60" height="28" rx="14" class="illo-pill"/>
                   <text x="250" y="183" text-anchor="middle" class="illo-pill-text">CPU</text>
                   <rect x="290" y="164" width="72" height="28" rx="14" class="illo-pill"/>
                   <text x="326" y="183" text-anchor="middle" class="illo-pill-text">CUDA</text>
                   <rect x="372" y="164" width="140" height="28" rx="14" class="illo-pill"/>
                   <text x="442" y="183" text-anchor="middle" class="illo-pill-text">no Python needed</text>
                 </svg>
               </div>

            .. code-block:: python

               torch.onnx.export(kornia.color.RgbToGrayscale(), x, "gray.onnx", dynamo=False)

            **Export once, run anywhere ONNX Runtime does** — chained with Hub-hosted operators into one graph.

            :doc:`See details <get-started/onnx>` :octicon:`arrow-right`

.. raw:: html

   <p class="kornia-sponsor-notice">Considering sponsoring? —
   <a href="community/sponsor.html">Inquire now</a></p>

Features
--------

.. rst-class:: kornia-section-lead

|count-operators-floor|\ + differentiable operators — functions and ``nn.Module`` layers, batched and
device-agnostic — across geometry, feature matching, filtering, color, augmentation, losses and pretrained
models. Six places to start below; the full map is the :doc:`API reference <api>`.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: kornia-cards kornia-gallery

   .. grid-item-card:: Warp, register, reconstruct
      :img-top: _static/img/warp_affine.png
      :link: geometry
      :link-type: doc

      Cameras, homographies, Lie groups, RANSAC, depth-to-3D — every transform differentiable, so a
      pose is something you optimise.

      +++
      |count-geometry| functions · ``kornia.geometry`` :octicon:`arrow-right`

   .. grid-item-card:: Match images
      :img-top: https://raw.githubusercontent.com/kornia/data/main/matching/matching_loftr.jpg
      :link: feature
      :link-type: doc

      Detect with KeyNet, DISK or ALIKED, match with LightGlue or LoFTR — classical SIFT and HardNet are
      there too.

      +++
      |count-feature| detectors & matchers · ``kornia.feature`` :octicon:`arrow-right`

   .. grid-item-card:: Filter and detect edges
      :img-top: _static/img/canny.png
      :link: filters
      :link-type: doc

      Canny, Sobel, Gaussian, bilateral and morphology, batched and differentiable — usable as a layer
      or inside a loss.

      +++
      |count-filters| operators · ``kornia.filters`` :octicon:`arrow-right`

   .. grid-item-card:: Augment everything
      :img-top: _static/img/AugmentationSequential.png
      :link: augmentation
      :link-type: doc

      One sampled transform applied to image, mask, boxes and keypoints alike — on the GPU, in 2D or 3D.

      +++
      |count-augmentation| transforms · ``kornia.augmentation`` :octicon:`arrow-right`

   .. grid-item-card:: Detect and segment with pretrained models
      :img-top: _static/img/face_detection.png
      :link: models/index
      :link-type: doc

      YuNet faces, RT-DETR objects, SAM masks, DexiNed edges — one line to build, weights on first use.

      +++
      |count-models| model families · Model zoo :octicon:`arrow-right`

   .. grid-item-card:: Recolor and enhance
      :img-top: _static/img/equalize_clahe.png
      :link: enhance
      :link-type: doc

      Lab, HSV, YUV and color maps; CLAHE, histogram equalisation, gamma and contrast — all trainable.

      +++
      ``kornia.color`` · ``kornia.enhance`` :octicon:`arrow-right`

.. rst-class:: kornia-learn-more

Prefer to learn by example? :doc:`Image matching <applications/image_matching>`,
:doc:`registration <applications/image_registration>`, :doc:`stitching <applications/image_stitching>`,
:doc:`denoising <applications/image_denoising>`, :doc:`face detection <applications/face_detection>` and
:doc:`visual prompting <applications/visual_prompting>` each walk through one task end to end — and
:doc:`Conventions & pitfalls <get-started/conventions>` is the one page to read before writing code.

.. raw:: html

   <div class="kornia-linkboard">
     <div>
       <p class="kornia-linkboard__title">Docs</p>
       <a href="applications/image_augmentations.html">Quick Start</a>
       <a href="https://www.kornia.org/tutorials/">Tutorial</a>
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
       <a href="community/adoption.html">Adoption</a>
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
