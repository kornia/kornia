.. _dexined_model:

DexiNed
=======

.. meta::
   :description: Detect thin image edges with DexiNed, Kornia's pretrained PyTorch edge-detection model.

.. rst-class:: kornia-badges

:bdg-primary:`Edge detection` :bdg-secondary:`MIT`

DexiNed is a dense inception-style network that predicts thin, well-localised edges and was trained from scratch
on the BIPED edge dataset. The quickest way to run it is
:class:`~kornia.contrib.edge_detection.EdgeDetectorBuilder`, which wraps :class:`~kornia.models.dexined.DexiNed`
with the resize, normalisation and sigmoid it expects and resizes the edge map back to the input size.

Run it
------

.. code-block:: python

    import torch
    from kornia.contrib.edge_detection import EdgeDetectorBuilder
    from kornia.io import load_image

    image = load_image("girona.png")[None]  # (1, 3, H, W) float in [0, 1]

    detector = EdgeDetectorBuilder.build("dexined", pretrained=True, image_size=352)  # runs the net at 352 px
    with torch.no_grad():
        edges = detector(image)[0]  # (1, 1, H, W) edge probabilities in [0, 1], one per input image

.. figure:: /_static/img/models/dexined.jpg
   :align: center
   :alt: A street photo and, next to it, the DexiNed edge map drawn as dark lines on white.

   The input image and its DexiNed edge probability map (dark = edge).

The detector accepts a batch tensor or a list of ``(3, H, W)`` images of different sizes and returns one edge map
per image. To use the bare network, :class:`~kornia.models.dexined.DexiNed` takes a ``(B, 3, H, W)`` input scaled to
``[0, 255]`` and mean-subtracted and returns the fused edge logits; ``EdgeDetectorBuilder`` also exposes
``to_onnx()`` for export.

Paper
-----

.. card::
    :link: https://www.computer.org/csdl/proceedings-article/wacv/2020/09093290/1jPbjFHmwi4

    **Dense Extreme Inception Network for Edge Detection**
    ^^^
    **Abstract:** Edge detection is the basis of many computer vision applications. State of the art predominantly relies on deep learning with two decisive factors: dataset content and network's architecture. Most of the publicly available datasets are not curated for edge detection tasks. Here, we offer a solution to this constraint. First, we argue that edges, contours and boundaries, despite their overlaps, are three distinct visual features requiring separate benchmark datasets. To this end, we present a new dataset of edges. Second, we propose a novel architecture, termed Dense Extreme Inception Network for Edge Detection (DexiNed), that can be trained from scratch without any pre-trained weights. DexiNed outperforms other algorithms in the presented dataset. It also generalizes well to other datasets without any fine-tuning. The higher quality of DexiNed is also perceptually evident thanks to the sharper and finer edges it outputs.

    **Tasks:** Edge Detection

    **Datasets:** BSD500, BIPED, MDBD

    **Journal:** 2020 IEEE Winter Conference on Applications of Computer Vision (WACV)

    **Licence:** MIT

    +++
    **Authors:** X. Soria and E. Riba and A. Sappa

.. image:: https://github.com/xavysp/DexiNed/raw/master/figs/DexiNed_banner.png
   :align: center
