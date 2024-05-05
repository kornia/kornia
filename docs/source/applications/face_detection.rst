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

The following Python example demonstrates how to use our API for detecting faces in images:

.. code-block:: python

    # select the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load and preprocess the image
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    img_raw = scale_image(img_raw, args.image_size)
    img = K.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    # create the detector and find the faces !
    face_detection = FaceDetector().to(device)

    with torch.no_grad():
        dets = face_detection(img)
    dets = [FaceDetectorResult(o) for o in dets[0]]


ONNX Model Export and Inference
-------------------------------

To use the face detection model with ONNX, you need to export the PyTorch model to the ONNX format:

.. code-block:: python

    face_detection.eval()
    fake_input = torch.randn(1, 3, 320, 320)
    onnx_path = "./face_detector.onnx"
    torch.onnx.export(face_detection, fake_input, onnx_path, verbose=True, opset_version=11)

Inference with ONNX model:

.. code-block:: python

    import onnxruntime as ort
    import numpy as np
    import kornia as K

    # Load ONNX model and create an inference session
    onnx_model = "./face_detector.onnx"
    ort_session = ort.InferenceSession(onnx_model)

    # Prepare the image
    img = img.astype(np.float32)[np.newaxis, ...]
    img = img.transpose(0, 3, 1, 2)  # Change data layout from NHWC to NCHW

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    dets = ort_session.run(None, ort_inputs)[0]

    # Convert detections to FaceDetectorResult
    dets = [FaceDetectorResult(torch.from_numpy(o)) for o in dets]

PS: You should be able to export the neural net directly (skipping the detector) for a faster ONNX model. You will have to handle the post-processing yourself.

.. code-block:: python

    face_detection.eval()
    fake_input = torch.randn(1, 3, 320, 320)
    onnx_path = "./face_detector.onnx"
    torch.onnx.export(face_detection.model, fake_input, onnx_path, verbose=True, opset_version=11)

Explore the capabilities of the face detector with this `tutorial <https://kornia.github.io/tutorials/nbs/face_detection.html>`_.
