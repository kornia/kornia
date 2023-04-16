kornia.contrib
==============

.. currentmodule:: kornia.contrib

Models
------

Base
^^^^
.. autoclass:: kornia.contrib.models.base.ModelBase
    :members:
    :undoc-members:

Structures
^^^^^^^^^^
.. _anchor SegmentationResults:
.. autoclass:: kornia.contrib.models.SegmentationResults
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.models.Prompts
    :members:
    :undoc-members:

ImagePrompter
-------------

.. autoclass:: kornia.contrib.prompter.ImagePrompter
    :members: set_image, reset_image, compile, predict, preprocess_image, preprocess_prompts

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

.. autoclass:: kornia.contrib.sam.SamModelType
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.sam.SamConfig
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.sam.Sam
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

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
