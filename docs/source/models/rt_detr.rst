Real-Time Detection Transformer (RT-DETR)
=========================================

.. code-block:: python

    from kornia.io import load_image
    from kornia.contrib.object_detection import ObjectDetectorBuilder

    input_img = load_image(img_path)[None]  # Load image to BCHW

    # NOTE: available models: 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
    # NOTE: recommended image scales: [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    detector = ObjectDetectorBuilder.build("rtdetr_r18vd", image_size=640)

    # get the output boxes
    boxes = detector(input_img)

    # draw the bounding boxes on the images directly.
    output = detector.draw(input_img, output_type="pil")
    output[0].save("Kornia-RTDETR-output.png")

    # convert the whole model to ONNX directly
    detector.to_onnx("RTDETR-640.onnx", image_size=640)


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
