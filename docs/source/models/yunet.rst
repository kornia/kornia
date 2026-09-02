.. _yunet_model:

YuNet
=====

.. rst-class:: kornia-badges

:bdg-primary:`Face detection` :bdg-secondary:`Apache-2.0`

YuNet is a millisecond-level face detector that predicts a box, a confidence and five facial landmarks (eyes, nose,
mouth corners) per face. Kornia exposes it as :class:`~kornia.contrib.FaceDetector`, which takes a ``(B, 3, H, W)``
image with pixel values in ``[0, 255]`` and returns one ``(N, 15)`` tensor per image; wrap each row in
:class:`~kornia.contrib.FaceDetectorResult` to read the fields by name.

Run it
------

.. code-block:: python

    from kornia.io import load_image
    from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint

    image = load_image("crowd.jpg")[None] * 255.0  # (1, 3, H, W); YuNet expects pixel values in [0, 255]

    detector = FaceDetector()  # downloads the weights on first use
    faces = [FaceDetectorResult(f) for f in detector(image)[0]]  # one result per detected face

    for face in faces:
        print(face.score, face.xmin, face.ymin, face.xmax, face.ymax, face.get_keypoint(FaceKeypoint.NOSE))

.. figure:: /_static/img/models/yunet.jpg
   :align: center
   :alt: A crowd photo (left) and the same photo with a green box and five yellow landmarks on each detected face (right).

   Boxes and landmarks returned by ``FaceDetector()`` with its defaults (``confidence_threshold=0.3``,
   ``nms_threshold=0.3``) on a 900×587 crowd image.

``FaceDetector(confidence_threshold=..., nms_threshold=..., top_k=..., keep_top_k=...)`` tunes the post-processing.
See :doc:`the face detection application page </applications/face_detection>` for a complete script that draws the results.

Paper
-----

.. card::
    :link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9429909

    **A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization**
    ^^^
    **Abstract:** Four-variable-independent-regression localization losses, such as Smooth- l 1 Loss, are used by default in modern detectors. Nevertheless, this kind of loss is oversimplified so that it is inconsistent with the final evaluation metric, intersection over union (IoU). Directly employing the standard IoU is also not infeasible, since the constant-zero plateau in the case of non-overlapping boxes and the non-zero gradient at the minimum may make it not trainable. Accordingly, we propose a systematic method to address these problems. Firstly, we propose a new metric, the extended IoU (EIoU), which is well-defined when two boxes are not overlapping and reduced to the standard IoU when overlapping. Secondly, we present the convexification technique (CT) to construct a loss on the basis of EIoU, which can guarantee the gradient at the minimum to be zero. Thirdly, we propose a steady optimization technique (SOT) to make the fractional EIoU loss approaching the minimum more steadily and smoothly. Fourthly, to fully exploit the capability of the EIoU based loss, we introduce an interrelated IoU-predicting head to further boost localization accuracy. With the proposed contributions, the new method incorporated into Faster R-CNN with ResNet50+FPN as the backbone yields 4.2 mAP gain on VOC2007 and 2.3 mAP gain on COCO2017 over the baseline Smooth- l 1 Loss, at almost no training and inferencing computational cost. Specifically, the stricter the metric is, the more notable the gain is, improving 8.2 mAP on VOC2007 and 5.4 mAP on COCO2017 at metric AP.

    **Tasks:** Face Detection

    **Datasets:** WIDER Face

    **Journal:** IEEE Transactions on Image Processing 2021

    **Licence:** Apache-2.0

    +++
    **Authors:** Hanyang Peng and Shiqi Yu
