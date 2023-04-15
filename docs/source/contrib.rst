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

Prompters
^^^^^^^^^
.. autoclass:: kornia.contrib.models.prompters.base.PrompterModelBase
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.models.prompters.image.ImagePrompter
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

.. autoclass:: SamModelType
    :members:
    :undoc-members:

.. autoclass:: Sam
    :members: build, from_pretrained, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.contrib.sam.prompter.SamPrompter
    :members: set_image, preprocess_image, preprocess_prompts, predict
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
