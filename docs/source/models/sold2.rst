SOLD2
=====

.. meta::
   :description: Detect and match line segments with SOLD², Kornia's pretrained PyTorch line-feature model.

.. rst-class:: kornia-badges

:bdg-primary:`Line detection` :bdg-primary:`Line matching` :bdg-secondary:`MIT`

SOLD² detects line segments and describes them in one network, so lines can be matched across views the way
keypoints are. :class:`~kornia.feature.SOLD2` returns the segments of each image together with a dense descriptor
map, and its ``match`` method pairs the segments of two images. :class:`~kornia.feature.SOLD2_detector` is the
detector alone.

Run it
------

.. code-block:: python

    import torch
    from kornia.color import rgb_to_grayscale
    from kornia.feature import SOLD2
    from kornia.io import load_image

    img1 = load_image("church_1.png")[None]  # (1, 3, H, W) float in [0, 1]
    img2 = load_image("church_2.png")[None]
    gray = rgb_to_grayscale(torch.cat([img1, img2]))  # (2, 1, H, W)

    sold2 = SOLD2(pretrained=True).eval()
    with torch.no_grad():
        out = sold2(gray)
        lines1, lines2 = out["line_segments"]  # list of (N, 2, 2) endpoints in (y, x) pixels, one per image
        desc = out["dense_desc"]  # (2, 128, H/4, W/4)
        matches = sold2.match(lines1, lines2, desc[0:1], desc[1:2])  # (N1,) index into lines2, -1 = no match

    matched1, matched2 = lines1[matches != -1], lines2[matches[matches != -1]]

.. figure:: /_static/img/models/sold2.jpg
   :align: center
   :alt: Two photographs of the same church with the detected line segments drawn in green, and a third panel where matched lines are drawn in the same colour on both images.

   Segments detected in each image (left, centre) and the matched pairs, one colour per pair (right).

The input is a grayscale batch; ``out`` also carries the raw ``junction_heatmap`` and ``line_heatmap`` maps. The
detection and matching thresholds are set through the ``config`` argument of :class:`~kornia.feature.SOLD2`.

Paper
-----

.. card::
    :link: https://arxiv.org/abs/2104.03362

    **SOLD²: Self-supervised Occlusion-aware Line Description and Detection**
    ^^^
    **Abstract:** Compared to feature point detection and description, detecting and matching line segments offer additional challenges. Yet, line features represent a promising complement to points for multi-view tasks. Lines are indeed well-defined by the image gradient, frequently appear even in poorly textured areas and offer robust structural cues. We thus hereby introduce the first joint detection and description of line segments in a single deep network. Thanks to a self-supervised training, our method does not require any annotated line labels and can therefore generalize to any dataset. Our detector offers repeatable and accurate localization of line segments in images, departing from the wireframe parsing approach. Leveraging the recent progresses in descriptor learning, our proposed line descriptor is highly discriminative, while remaining robust to viewpoint changes and occlusions. We evaluate our approach against previous line detection and description methods on several multi-view datasets created with homographic warps as well as real-world viewpoint changes. Our full pipeline yields higher repeatability, localization accuracy and matching metrics, and thus represents a first step to bridge the gap with learned feature points methods.

    **Tasks:** Line detection, Line description, Line matching

    **Datasets:** Wireframe, YorkUrban, ETH3D

    **Conference:** CVPR 2021

    **Licence:** MIT

    +++
    **Authors:** Rémi Pautrat*, Juan-Ting Lin*, Viktor Larsson, Martin R. Oswald, Marc Pollefeys

.. image:: https://github.com/cvg/SOLD2/raw/main/assets/videos/demo_moving_camera.gif
   :align: center
