LoFTR
=====

.. rst-class:: kornia-badges

:bdg-primary:`Image matching` :bdg-secondary:`Apache-2.0`

LoFTR matches two images without detecting keypoints first: a transformer establishes coarse pixel-wise matches
and refines them at a finer level, which is what makes it work on low-texture surfaces where detectors fail.
:class:`~kornia.feature.LoFTR` takes a dictionary with two grayscale images and returns the matched pixel
coordinates and a confidence per match.

Run it
------

.. code-block:: python

    import torch
    from kornia.color import rgb_to_grayscale
    from kornia.feature import LoFTR
    from kornia.io import load_image

    img1 = load_image("church_1.png")[None]  # (1, 3, H, W) float in [0, 1]
    img2 = load_image("church_2.png")[None]

    matcher = LoFTR(pretrained="outdoor").eval()  # 'outdoor' (MegaDepth) or 'indoor' (ScanNet)
    with torch.no_grad():
        out = matcher({"image0": rgb_to_grayscale(img1), "image1": rgb_to_grayscale(img2)})

    pts1, pts2, conf = out["keypoints0"], out["keypoints1"], out["confidence"]  # (N, 2) (x, y) pixels, (N, 2), (N,)

.. figure:: /_static/img/models/loftr.jpg
   :align: center
   :alt: Two photographs of the same church side by side, with green lines connecting the LoFTR matches between them.

   Matches between two views of the same building; each green line joins ``keypoints0[i]`` to ``keypoints1[i]``.
   Feed the pairs to :class:`~kornia.geometry.ransac.RANSAC` or :func:`~kornia.geometry.homography.find_homography_dlt`
   to estimate the geometry between the views.

The two images may have different sizes, and ``out["batch_indexes"]`` tells which pair of a batch each match
belongs to. :class:`~kornia.feature.LocalFeatureMatcher` and the :doc:`image matching application
</applications/image_matching>` show LoFTR next to the detector-based pipelines.

Paper
-----

.. card::
    :link: https://paperswithcode.com/paper/loftr-detector-free-local-feature-matching

    **LoFTR: Detector-Free Local Feature Matching with Transformers**
    ^^^
    **Abstract:** We present a novel method for local image feature matching. Instead of performing image feature detection, description, and matching sequentially, we propose to first establish pixel-wise dense matches at a coarse level and later refine the good matches at a fine level. In contrast to dense methods that use a cost volume to search correspondences, we use self and cross attention layers in Transformer to obtain feature descriptors that are conditioned on both images. The global receptive field provided by Transformer enables our method to produce dense matches in low-texture areas, where feature detectors usually struggle to produce repeatable interest points. The experiments on indoor and outdoor datasets show that LoFTR outperforms state-of-the-art methods by a large margin. LoFTR also ranks first on two public benchmarks of visual localization among the published methods.

    **Tasks:** Local Feature Matching, Visual Localisation

    **Datasets:** ScanNet, HPatches, MegaDepth, InLoc

    **Conference:** CVPR 2021

    **Licence:** Apache-2.0

    +++
    **Authors:** Jiaming Sun*, Zehong Shen*, Yu'ang Wang*, Hujun Bao, Xiaowei Zhou

.. image:: https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/loftr-github-demo.gif
   :align: center
