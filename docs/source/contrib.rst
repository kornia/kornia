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

EfficientViT
^^^^^^^^^^^^

.. autoclass:: kornia.contrib.models.efficient_vit.EfficientViT
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,


.. autoclass:: kornia.contrib.models.efficient_vit.EfficientViTConfig
    :members:
    :undoc-members:

Backbones
^^^^^^^^^

.. autoclass:: kornia.contrib.models.efficient_vit.backbone.EfficientViTBackbone
    :members:
    :undoc-members:

.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_b0
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_b1
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_b2
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_b3

.. autoclass:: kornia.contrib.models.efficient_vit.backbone.EfficientViTLargeBackbone
    :members:
    :undoc-members:

.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_l0
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_l1
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_l2
.. autofunction:: kornia.contrib.models.efficient_vit.backbone.efficientvit_backbone_l3

Structures
^^^^^^^^^^

.. _anchor SegmentationResults:
.. autoclass:: kornia.contrib.models.SegmentationResults
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.models.Prompts
    :members:
    :undoc-members:

VisualPrompter
--------------

.. autoclass:: kornia.contrib.visual_prompter.VisualPrompter
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

Object Detection
----------------

.. autoclass:: kornia.contrib.object_detection.BoundingBoxDataFormat
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.contrib.object_detection.BoundingBox
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.object_detection.ObjectDetectorResult
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.object_detection.ObjectDetector
    :members:
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.contrib.object_detection.ResizePreProcessor
    :members:
    :undoc-members:

.. autofunction:: kornia.contrib.object_detection.results_from_detections

Real-Time Detection Transformer (RT-DETR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: kornia.contrib.models.rt_detr.RTDETRModelType
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.contrib.models.rt_detr.RTDETRConfig
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.models.rt_detr.RTDETR
    :members: from_config, forward, load_checkpoint
    :undoc-members:
    :special-members: __init__,

.. autoclass:: kornia.contrib.models.rt_detr.DETRPostProcessor
    :members:
    :undoc-members:

Image Segmentation
------------------
.. autofunction:: connected_components

Segment Anything (SAM)
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: kornia.contrib.models.sam.SamModelType
    :members:
    :undoc-members:
    :member-order: bysource

.. autoclass:: kornia.contrib.models.sam.SamConfig
    :members:
    :undoc-members:

.. autoclass:: kornia.contrib.models.sam.Sam
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
.. autoclass:: TinyViT
    :members:
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
