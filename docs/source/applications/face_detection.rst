Face Detection
==============

.. image:: https://github.com/ShiqiYu/libfacedetection/raw/master/images/cnnresult.png
   :align: right
   :width: 20%

Face detection is the task of detecting faces in a photo or video (and distinguishing them from other objects).
We provide the :py:class:`kornia.contrib.FaceDetector` to perform multi-face detection in real-time using the
:ref:`yunet_model` model.

Learn more: `https://paperswithcode.com/task/face-detection <https://paperswithcode.com/task/face-detection>`_

..  youtube:: hzQroGp5FSQ

Using our API you easily detect faces in images as shown below:

.. code-block:: python

    # select the device
    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load the image and scale
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    img_raw = scale_image(img_raw, args.image_size)

    # preprocess
    img = K.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    # create the detector and find the faces !
    face_detection = FaceDetector().to(device)

    with torch.no_grad():
        dets = face_detection(img)
    dets = [FaceDetectorResult(o) for o in dets[0]]


Play yourself with the detector and generate new images with this `tutorial <https://kornia.github.io/tutorials/nbs/face_detection.html>`_.
