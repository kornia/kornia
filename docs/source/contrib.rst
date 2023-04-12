kornia.contrib
==============

.. currentmodule:: kornia.contrib

Edge Detection
--------------

.. autoclass:: EdgeDetector

Face Detection
--------------

.. autoclass:: FaceDetector

.. autoclass:: FaceKeypoint
    :members:
    :undoc-members:

.. autoclass:: FaceDetectorResult
    :members:
    :undoc-members:

Interactive Demo
^^^^^^^^^^^^^^^^
.. raw:: html

    <gradio-app space="kornia/Face-Detection"></gradio-app>

Visit the `Kornia face detection demo on the Hugging Face Spaces
<https://huggingface.co/spaces/kornia/Face-Detection>`_.

Image Segmentation
------------------
.. autofunction:: connected_components


Segment Anything (SAM)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Sam
    :members: build, from_pretrained, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

.. autoclass:: SamModelType
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.sam.predictor.SamPredictor
    :members:
    :special-members: __call__,

.. autoclass:: kornia.contrib.sam.base.SegmentationResults
    :members:



Image Patches
-------------

.. autofunction:: compute_padding
.. autofunction:: extract_tensor_patches
.. autofunction:: combine_tensor_patches

.. autoclass:: ExtractTensorPatches
.. autoclass:: CombineTensorPatches

Image Classification
--------------------

.. autoclass:: VisionTransformer
.. autoclass:: MobileViT
.. autoclass:: ClassificationHead

Image Stitching
---------------

.. autoclass:: ImageStitcher

Lambda
------

.. autoclass:: Lambda

Distance Transform
------------------

.. autofunction:: distance_transform
.. autofunction:: diamond_square

.. autoclass:: DistanceTransform
