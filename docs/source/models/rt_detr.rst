RT-DETR
=======

.. rst-class:: kornia-badges

:bdg-primary:`Object detection` :bdg-secondary:`Apache-2.0`

RT-DETR is a real-time, end-to-end transformer detector trained on MS-COCO. Kornia wraps it in
:class:`~kornia.contrib.object_detection.RTDETRDetectorBuilder`, which bundles the resize/normalise pre-processing
and the box post-processing, so a ``(B, 3, H, W)`` float image in ``[0, 1]`` goes in and one ``(D, 6)`` tensor per
image comes out with ``class_id, score, x, y, w, h`` in the original pixel coordinates.

Run it
------

.. code-block:: python

    from kornia.io import load_image
    from kornia.contrib.object_detection import RTDETRDetectorBuilder

    image = load_image("delorean.png")[None]  # (1, 3, H, W) float in [0, 1]

    detector = RTDETRDetectorBuilder.build("rtdetr_r18vd", image_size=640)  # downloads the COCO weights
    detections = detector(image)  # list with one (D, 6) tensor per image: class id, score, x, y, w, h

    for class_id, score, x, y, w, h in detections[0].tolist():
        print(f"class {int(class_id)}: {score:.2f} at ({x:.0f}, {y:.0f}) size {w:.0f}x{h:.0f}")

.. figure:: /_static/img/models/rt_detr.jpg
   :align: center
   :alt: A photo of a car (left) and the same photo with two RT-DETR detections drawn as green boxes with class id and score (right).

   Input image and the boxes returned by ``rtdetr_r18vd`` at ``image_size=640``, drawn with the detected class id
   (COCO id 2 is *car*) and confidence. Figures on these pages are rendered by ``docs/generate_model_examples.py``.

Detections are filtered with ``confidence_threshold=0.3`` by default; pass a different value to ``build`` to keep
more or fewer boxes. ``detector.visualize(image, detections, output_type="pil")`` draws the boxes for you, and
``detector.to_onnx("rtdetr-640.onnx", image_size=640)`` exports the model together with its pre- and post-processing.

Available variants, from fastest to most accurate: ``rtdetr_r18vd``, ``rtdetr_r34vd``, ``rtdetr_r50vd_m``,
``rtdetr_r50vd`` and ``rtdetr_r101vd``. Recommended input scales are multiples of 32 between 480 and 800; ``640`` is
the value the weights were trained for.

Paper
-----

.. card::
    :link: https://arxiv.org/abs/2304.08069

    **RT-DETR**
    ^^^
    **Abstract:** Recently, end-to-end transformer-based detectors (DETRs) have achieved remarkable performance.
    However, the issue of the high computational cost of DETRs has not been effectively addressed, limiting their
    practical application and preventing them from fully exploiting the benefits of no post-processing, such as
    non-maximum suppression (NMS). In this paper, we first analyze the influence of NMS in modern real-time object
    detectors on inference speed, and establish an end-to-end speed benchmark. To avoid the inference delay caused
    by NMS, we propose a Real-Time DEtection TRansformer (RT-DETR), the first real-time end-to-end object detector
    to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale
    features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection
    to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment
    of the inference speed by using different decoder layers without the need for retraining, which facilitates the
    practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS
    on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in
    both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming
    DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. Source code and pretrained models
    will be available at PaddleDetection.

    **Tasks:** Detection

    **Datasets:** MS-COCO

    **Licence:** Apache 2.0

    +++
    **Authors:** Wenyu Lv, Shangliang Xu, Yian Zhao, Guanzhong Wang, Jinman Wei, Cheng Cui, Yuning Du, Qingqing Dang, Yi Liu
