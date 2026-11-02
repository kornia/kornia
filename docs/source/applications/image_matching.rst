Image Matching
==============

.. meta::
   :description: Match images with Kornia's PyTorch local features, LoFTR, geometric estimation and RANSAC tools.

Image matching is a process of finding pixel and region correspondences between two images of the same scene.
Such correspondences are useful for 3D reconstruction of the scene and relative camera pose estimation.
It is also known as "wide baseline stereo"; you can read more about it on the `Wide Baseline Stereo Blog <https://ducha-aiki.github.io/wide-baseline-stereo-blog/2021/01/09/wxbs-in-simple-terms.html>`_.

We provide many modules and functions for image matching, from building blocks like
:doc:`local feature detectors, descriptors and descriptor matchers </feature>` to
:doc:`geometric model estimation </geometry.epipolar>` and :doc:`RANSAC </geometry.ransac>`.

However, we recommend starting with a high-level API such as :py:class:`~kornia.feature.LoFTR`, which finds
correspondences between two grayscale images in a single call:

.. code:: python

    import torch
    from kornia.feature import LoFTR

    img1 = torch.rand(1, 1, 480, 640)  # (B, 1, H, W) grayscale images in [0, 1]
    img2 = torch.rand(1, 1, 480, 640)

    matcher = LoFTR(pretrained="outdoor")
    with torch.inference_mode():
        correspondences = matcher({"image0": img1, "image1": img2})

    keypoints0 = correspondences["keypoints0"]  # (N, 2) points in img1, (x, y) pixel coordinates
    keypoints1 = correspondences["keypoints1"]  # (N, 2) matching points in img2
    confidence = correspondences["confidence"]  # (N,)

.. image:: https://raw.githubusercontent.com/kornia/data/main/matching/matching_loftr.jpg
   :alt: LoFTR correspondences between two images

You can also go through our full tutorial using Colab, found `here <https://www.kornia.org/tutorials/nbs/image_matching.html>`_.

Which model should I use?
-------------------------

Kornia's local-feature stack has three kinds of components, all in :doc:`kornia.feature </feature>`:

1. **Detectors** find keypoints: classical responses (Harris, GFTT, Hessian, DoG) and learned detectors (KeyNet, DISK, DeDoDe, ALIKED).
2. **Descriptors** describe the patch around each keypoint: SIFT, MKD, HardNet, HyNet, SOSNet, TFeat.
3. **Matchers** pair descriptors across images: nearest-neighbour and ratio tests, the handcrafted AdaLAM, the
   learned LightGlue, and LoFTR, which matches densely without a detector at all.

The table below shows their performance on the `IMC2021 benchmark
<https://www.cs.ubc.ca/research/image-matching-challenge/2021/leaderboard/>`_ (mAA at 10 degrees).

.. list-table:: IMC2021 Benchmark, 8000 features
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - Feature name
     - Stereo mAA @ 10 degrees, PhotoTourism.
     - Multiview mAA @ 10 degrees, PhotoTourism.
     - Stereo mAA @ 10 degrees, PragueParks.
     - Multiview mAA @ 10 degrees, PragueParks.
   * - DISK-LightGlue
     - 0.6184
     - 0.7741
     - 0.6116
     - 0.4988
   * - LoFTR
     - 0.6090
     - 0.7609
     - 0.7546
     - 0.4711
   * - OpenCV-DoG-HardNet-LightGlue
     - 0.5850
     - 0.7587
     - 0.6525
     - 0.4973
   * - OpenCV-DoG-AffNet-HardNet8-AdaLAM
     - 0.5502
     - 0.7522
     - 0.5998
     - 0.4712
   * - Upright SIFT (OpenCV)
     - 0.5122
     - 0.6849
     - 0.6060
     - 0.4439

.. list-table:: IMC2021 Benchmark, 2048 features
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - Feature name
     - Stereo mAA @ 10 degrees, PhotoTourism.
     - Multiview mAA @ 10 degrees, PhotoTourism.
     - Stereo mAA @ 10 degrees, PragueParks.
     - Multiview mAA @ 10 degrees, PragueParks.
   * - DISK-LightGlue
     - 0.5720
     - 0.7543
     - 0.5099
     - 0.4565
   * - OpenCV-DoG-HardNet-LightGlue
     - 0.3954
     - 0.6272
     - 0.5157
     - 0.4456
   * - Upright SIFT (OpenCV)
     - 0.3827
     - 0.5545
     - 0.4136
     - 0.3607

Rules of thumb: :class:`~kornia.feature.LoFTR` works best for indoor scenes, whereas
:class:`~kornia.feature.DISK` + :class:`~kornia.feature.LightGlue` and :class:`~kornia.feature.DeDoDe` + LightGlue
work best outdoors. For domains far from natural photographs, e.g. remote sensing or medical imaging, SIFT or
SIFT + :class:`~kornia.feature.HardNet` + LightGlue are often the more robust choice. The next section compares
their speed and accuracy with the authors' code, device by device.

.. _feature-speed-by-device:

Speed and accuracy against the reference implementations
--------------------------------------------------------

Kornia's local-feature pipelines run at their authors' speed or faster on every device measured, and are as
accurate on every cohort measured. Choose differently in these cases:

- **ALIKED on CUDA: compile it** for repeated inference. Eager, it runs at 0.79× the authors' speed: the authors call
  torchvision's deformable-convolution kernel, while kornia, which does not depend on torchvision, uses a PyTorch one
  that launches many small kernels. Compiled, it runs at 1.63× at a fixed image size. On CPUs keep it eager: it runs
  at 1.7–2.0× the authors' speed, and compiling it made it slower on the x86 CPU.
- **SIFT on a CPU: use OpenCV's**, which is 8–14× faster than kornia's on the Apple M1 CPU. Use kornia's SIFT when
  the images are on a GPU, or you need gradients or batches: :class:`~kornia.feature.SIFTFeature` takes one image per
  call, :class:`~kornia.feature.SIFTFeatureScaleSpace` a batch.
- **Ratio-test matching on CUDA or the Apple M1 CPU: use faiss.** Its exact k-nearest-neighbour search returns the
  same matches as :func:`~kornia.feature.match_snn` and :func:`~kornia.feature.match_smnn`, up to 2.2× faster on
  CUDA and up to 2.0× faster on the M1 CPU. On the x86 CPU, kornia is the fastest.
- **XFeat\* is more accurate on planar scenes, on purpose.** :meth:`~kornia.feature.XFeat.match_xfeat_star` departs
  from the authors' refinement (see :class:`~kornia.feature.XFeat`): 36.09 % against 20.46 % on HPatches, and from
  2.7 points below to 0.3 above the authors' on PhotoTourism.

A speed figure is the reference's latency divided by kornia's, so a value above 1 means kornia is faster. The
reference is the authors' code, or OpenCV for SIFT. ``~`` marks a noisy figure: on the M1, a relative timing spread
above 10 %; on CUDA, minimum- and median-based ratios more than 3 % apart. ``—`` marks a cell that was not measured.
*Same* accuracy means within 2.5 percentage points, the weight of one Oxford pair, on every cohort measured.

.. list-table:: kornia against the reference implementations: speed ratio by device, and accuracy
   :header-rows: 1
   :widths: 26 10 10 10 18 26

   * - Pipeline (reference)
     - Apple M1 CPU
     - Apple M1 MPS
     - x86 CPU
     - CUDA
     - Accuracy
   * - :class:`~kornia.feature.ALIKED` n16 (authors' Python code)
     - **1.99×**
     - 0.99×
     - **1.66×**
     - ~0.79× eager; **1.63×** compiled, ~1.23× with ``dynamic=True``
     - Same
   * - :class:`~kornia.feature.XFeat` (authors)
     - **1.53×**
     - ~1.03×
     - **1.64×**
     - 0.98×
     - Same
   * - XFeat\*, :meth:`~kornia.feature.XFeat.match_xfeat_star` (authors)
     - 1.00×
     - ~1.06×
     - 0.98×
     - ~1.01×
     - Better on planar scenes, mixed on PhotoTourism
   * - :class:`~kornia.feature.DISK` (authors)
     - ~1.23×
     - 1.00×
     - 1.03×
     - 1.05×
     - Same
   * - DISK + :class:`~kornia.feature.LightGlue` (authors)
     - ~1.10×
     - 1.11×
     - **1.50×**
     - ~0.97×
     - Same
   * - :class:`~kornia.feature.LoFTR` (authors)
     - 0.99×
     - ~1.06×
     - 1.06×
     - **1.26×**
     - Same
   * - :class:`~kornia.feature.DeDoDe`, B descriptor (authors)
     - 1.01×
     - —
     - 0.96×
     - ~0.96×
     - Same
   * - DeDoDe, G descriptor (authors)
     - 1.09×, one pair
     - —
     - **1.16×**
     - **1.19×**
     - Same
   * - :class:`~kornia.feature.MKDDescriptor`, polar or concatenated (authors)
     - **2.33–2.45×**
     - **5.97–6.34×**
     - **7.04–8.50×**
     - ~1.14–1.30×
     - Same
   * - MKDDescriptor, Cartesian (authors)
     - ~1.54×
     - **2.93×**
     - **3.92×**
     - 0.96×
     - Same
   * - :class:`~kornia.feature.HyNet` on shared SIFT frames (authors)
     - 1.00×
     - **1.35×**
     - **1.21×**
     - 1.00×
     - Same
   * - :class:`~kornia.feature.HardNet`, :class:`~kornia.feature.SOSNet`, :class:`~kornia.feature.TFeat` on
       shared SIFT frames (authors)
     - 0.99–1.02×
     - 0.97–1.06×
     - **1.29–1.31×**
     - 0.99–1.01×
     - Same
   * - :func:`~kornia.feature.match_adalam` (authors)
     - 1.00×
     - 1.00×
     - —
     - —
     - Same
   * - :class:`~kornia.feature.SIFTFeature`, :class:`~kornia.feature.SIFTFeatureScaleSpace` (OpenCV RootSIFT)
     - **0.07–0.13×**
     - —
     - —
     - —
     - Same, Oxford only

**Matching.** :func:`~kornia.feature.match_nn`, :func:`~kornia.feature.match_mnn`,
:func:`~kornia.feature.match_snn` and :func:`~kornia.feature.match_smnn` were timed against OpenCV's ``BFMatcher``
and the exact k-nearest-neighbour search of faiss 1.15.1, on RootSIFT descriptors of PhotoTourism images, 1,024 to
8,192 per image. All three return the same matches.

- **x86 CPU**, four or fourteen threads: kornia is the fastest, at 1.4–4.8× the speed of OpenCV and 1.15–2.9× that
  of faiss.
- **CUDA**: faiss is 1.2–2.2× faster for ``match_snn``, and 1.3–1.6× for ``match_smnn`` except at 4,096
  descriptors, where the two tie. ``match_nn`` and ``match_mnn`` tie or favour kornia, except ``match_nn`` at 8,192
  descriptors (faiss 1.3×). At 2,048 descriptors each matcher takes under 1 ms per pair. OpenCV's pip build has no
  CUDA matcher.
- **Apple M1 CPU**, four threads: kornia runs at 2.1–3.5× the speed of OpenCV, and 1.9–2.4× that of faiss for
  ``match_nn`` and ``match_mnn``. faiss is 1.6–2.0× faster for ``match_snn`` and 1.3–2.0× for ``match_smnn``. faiss
  and kornia ran in separate processes, because each ships its own OpenMP runtime; OpenCV ran 3–14 % faster in
  faiss's process, so the faiss figures may favour faiss by that much.

**How these were measured.** Kornia's main branch between 30 October and 2 November 2026 ran against the authors'
code at pinned commits, with PyTorch 2.14.0. Images were resized to 640 pixels on the long side, rounded to
multiples of 32. Inference was float32, with a budget of 2,048 features; LoFTR is dense and uncapped. The ALIKED
reference is the authors' Python code; their compiled sampler extension was not timed.

- **Apple M1, 8 GB**, four CPU threads. A figure compares median timings of the full pipeline on Oxford pairs: both
  extractions, matching, a shared OpenCV MAGSAC estimate and scoring. That shared step pulls the ratios towards 1.
  Accuracy covers Oxford (40 pairs), HPatches (145 viewpoint pairs from 29 sequences) and PhotoTourism (130 pairs of
  the IMC 2021 validation set, from three scenes). The planar cohorts count the pairs whose mean homography corner
  error is below 3 px; PhotoTourism uses pose mAA over 1–10 degrees. Timing DeDoDe on MPS exhausted the M1's memory.
- **Intel i7-14700K and NVIDIA RTX 4090**, four CPU threads. A figure covers the model path on PhotoTourism pairs:
  both extractions, matching, and the transfer of the matches to the host, with no estimator. CPU figures compare
  median timings. CUDA rounds on that host were bimodal, so CUDA figures compare each pair's fastest round. Accuracy
  is pose mAA on the same 130 PhotoTourism pairs, with one matching recipe per pipeline on both sides, which need not
  be the M1 recipe. Compiled ALIKED compiles kornia's model only; the authors' runs eagerly. On CUDA, both DeDoDe
  implementations ran in their default float16 autocast.

A ratio compares two implementations on the same machine and pairs, never two machines, and need not carry over to
other hardware, image sizes or feature budgets. Pipelines whose two sides follow different recipes are left out,
because their gaps come from the recipes: the native KeyNet + HyNet, HesAffNet + HardNet and C4 recipes, and Harris +
HardNet against OpenCV's Harris. The full tables, per-pair spread and protocol are in
`pull request #6452 <https://github.com/kornia/kornia/pull/6452>`_.
