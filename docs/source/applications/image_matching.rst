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
SIFT + :class:`~kornia.feature.HardNet` + LightGlue are often the more robust choice. DeDoDe and speed benchmarks
are coming soon.
