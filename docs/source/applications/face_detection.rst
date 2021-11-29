Face Detection
==============

.. image:: https://github.com/ShiqiYu/libfacedetection/raw/master/images/cnnresult.png
   :align: right
   :width: 20%

Face detection is the task of detecting faces in a photo or video (and distinguishing them from other objects).
We provide the :py:class:`kornia.contrib.FaceDetector` to perform multi-face detection in real-time uding the
`YuNet` model.

Learn more: `https://paperswithcode.com/task/face-detection <https://paperswithcode.com/task/face-detection>`_

..  youtube:: hzQroGp5FSQ

Using our API you easily detect faces in images as shown below:

.. literalinclude:: ../../../examples/face_detection/main.py
   :language: python
   :lines: 36-53

Check the full examples `here <https://github.com/kornia/kornia/tree/master/examples/face_detection/main.py>`_

For a real-time example using the camera check this other `example <https://github.com/kornia/kornia/tree/master/examples/face_detection/main_video.py>`_