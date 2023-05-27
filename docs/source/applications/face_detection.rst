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

.. literalinclude:: ../../../examples/face_detection/main.py
   :language: python
   :lines: 36-54

Check the full example `here <https://github.com/kornia/kornia/tree/master/examples/face_detection/main.py>`_
or run a real-time application using the camera with this `example <https://github.com/kornia/kornia/tree/master/examples/face_detection/main_video.py>`_.

The Kornia AI Game
------------------

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/face_detection_7_1.png
        :width: 49 %
    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/face_detection_13_1.png
        :width: 49 %

.. tip::
   Play yourself with the detector and generate new images with this `tutorial <https://kornia-tutorials.readthedocs.io/en/latest/face_detection.html>`_.
