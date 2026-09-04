Face Detection
==============

.. meta::
   :description: Detect multiple faces in images and video with Kornia's real-time PyTorch FaceDetector and YuNet model.

.. image:: https://github.com/ShiqiYu/libfacedetection/raw/master/images/cnnresult.png
   :align: right
   :width: 20%

Face detection is the task of detecting faces in a photo or video (and distinguishing them from other objects).
We provide the :py:class:`kornia.contrib.FaceDetector` to perform multi-face detection in real-time using the
:ref:`yunet_model` model.

Learn more: `https://paperswithcode.com/task/face-detection <https://paperswithcode.com/task/face-detection>`_

..  youtube:: hzQroGp5FSQ

Using our API you can easily detect faces in images as shown below:

.. code-block:: python

    import torch
    import kornia as K
    from kornia.contrib import FaceDetector, FaceDetectorResult
    from kornia.io import ImageLoadType

    # select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the image as a (1, 3, H, W) float tensor in [0, 255], the range YuNet expects
    img = K.io.load_image("image.jpg", ImageLoadType.RGB32, device=device)[None] * 255.0

    # create the detector and find the faces
    face_detection = FaceDetector().to(device)

    with torch.no_grad():
        dets = face_detection(img)
    dets = [FaceDetectorResult(o) for o in dets[0]]

    for det in dets:
        print(det.score, det.top_left, det.bottom_right)

Play with the detector yourself and generate new images with this `tutorial <https://www.kornia.org/tutorials/nbs/face_detection.html>`_.
