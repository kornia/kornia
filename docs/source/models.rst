kornia.models
=============

.. meta::
   :description: The Kornia models overview provides detailed information about key built-in models for computer vision tasks, including real-time object detection (RT-DETR), edge detection (DexiNed), segmentation (UNet, DeepLabV3), and multi-object tracking (BoxMotTracker). It offers comprehensive documentation on each model, including methods, parameters, and example usage to streamline the integration of these models into computer vision workflows.


Builders for Kornia's ready-to-use models (object detection, edge detection, semantic segmentation, multi-object tracking and Kimi-VL).
Each builder returns a configured model with pretrained weights. For the papers behind the models, see the :doc:`Models </models/index>` section.
Pretrained weights are downloaded on first use, and every builder returns a regular ``nn.Module`` that accepts a batched
``(B, 3, H, W)`` float image in ``[0, 1]``. :func:`kornia.io.get_sample_images` provides a couple of sample images for quick experiments.

.. _RTDETRDetectorBuilder:

RTDETRDetectorBuilder
---------------------

The `RTDETRDetectorBuilder` class is a builder for constructing a detection model based on the RT-DETR architecture, which is designed for real-time object detection. It is capable of detecting multiple objects within an image and provides efficient inference suitable for real-world applications.

**Key Methods:**

- `build`: Constructs and returns an instance of the RTDETR detection model.
- `visualize`: Draws the detected boxes on the input images.

.. autoclass:: kornia.contrib.object_detection.RTDETRDetectorBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code demonstrates how to use `RTDETRDetectorBuilder` to detect objects in an image:

   .. code-block:: python

       import kornia
       from kornia.contrib.object_detection import RTDETRDetectorBuilder

       image = kornia.io.get_sample_images()[0][None]
       model = RTDETRDetectorBuilder.build()
       detections = model(image)  # list of (D, 6) tensors: class id, score, x, y, w, h
       drawn = model.visualize(image, detections)  # the boxes drawn on the image

.. _EdgeDetectorBuilder:

EdgeDetectorBuilder
-------------------

The `EdgeDetectorBuilder` class implements a state-of-the-art edge detection model based on DexiNed, which excels at detecting fine-grained edges in images. This model is well-suited for tasks like medical imaging, object contour detection, and more.

**Key Methods:**

- `build`: Builds and returns an instance of the DexiNed edge detection model.
- `visualize`: Returns the edge maps as images for further processing or display.

.. autoclass:: kornia.contrib.edge_detection.EdgeDetectorBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code shows how to use the `EdgeDetectorBuilder` to detect edges in an image:

   .. code-block:: python

       import kornia
       from kornia.contrib.edge_detection import EdgeDetectorBuilder

       image = kornia.io.get_sample_images()[0][None]
       model = EdgeDetectorBuilder.build()
       edges = model(image)  # list with one (1, 1, H, W) edge map per image

.. _SegmentationModels:

SegmentationModelsBuilder
-------------------------

The `SegmentationModelsBuilder` class offers a flexible API for building and running segmentation models from the
optional `segmentation_models_pytorch <https://github.com/qubvel-org/segmentation_models.pytorch>`_ package
(``pip install segmentation_models_pytorch``). It supports a variety of architectures such as UNet, FPN, DeepLabV3 and
others, with ImageNet-pretrained encoders.

**Key Methods:**

- `build`: Constructs a segmentation model for the chosen architecture, encoder and number of classes.

**Main parameters of** `build`:

- `model_name`: (str) Name of the segmentation architecture to use, e.g., `"Unet"`, `"DeepLabV3"`.
- `encoder_name`: (str) Name of the encoder backbone, e.g., `"resnet34"`.
- `classes`: (int) The number of output classes for segmentation.

.. autoclass:: kornia.models.segmentation.segmentation_models.SegmentationModelsBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   Here's an example of how to use `SegmentationModelsBuilder` for binary segmentation:

   .. code-block:: python

       import kornia
       from kornia.models.segmentation.segmentation_models import SegmentationModelsBuilder

       input_tensor = kornia.io.get_sample_images()[0][None]
       model = SegmentationModelsBuilder.build(model_name="Unet", encoder_name="resnet34", classes=1)
       segmented_output = model(input_tensor)
       print(segmented_output.shape)

.. _KimiVLBuilder:

KimiVLBuilder
-------------

The `KimiVLBuilder` class constructs Kimi-VL models from a configuration or downloads pretrained weights. Pretrained
loading currently supports only the converted Kimi-VL-A3B-Instruct vision encoder and projector checkpoint.

**Key Methods:**

- `from_config`: Constructs a randomly initialized Kimi-VL model from a `KimiVLConfig`.
- `from_pretrained_hf`: Downloads and strictly loads the supported pretrained checkpoint.

.. autoclass:: kornia.models.kimi_vl.KimiVLBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code loads the supported pretrained Kimi-VL vision model:

   .. code-block:: python

       from kornia.models.kimi_vl import KimiVLBuilder

       model = KimiVLBuilder.from_pretrained_hf().eval()

.. _BoxMotTracker:

BoxMotTracker
-------------

The `BoxMotTracker` class is used for multi-object tracking in video streams. It combines a Kornia object detector with
a tracker from the optional `boxmot <https://github.com/mikel-brostrom/boxmot>`_ package (``pip install boxmot``) to
track bounding boxes across frames.

**Key Methods:**

- `__init__`: Initializes the multi-object tracker from a detector and a tracker model name.
- `update`: Updates the tracker with a new image frame.
- `visualize`: Draws the tracked boxes (and optionally their trajectories) on a frame.

**Main parameters:**

- `detector`: (ObjectDetector | str) The object detector instance, or the name of an RT-DETR model to build, e.g. `"rtdetr_r18vd"`.
- `tracker_model_name`: (str) The boxmot tracker to use, e.g. `"DeepOCSORT"`.

.. autoclass:: kornia.contrib.boxmot_tracker.BoxMotTracker
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following example demonstrates how to track objects across multiple frames using `BoxMotTracker`:

   .. code-block:: python

       import kornia
       from kornia.contrib.boxmot_tracker import BoxMotTracker

       image = kornia.io.get_sample_images()[0][None]
       model = BoxMotTracker()
       for i in range(4):
           model.update(image)  # Update the tracker with new frames
       tracked = model.visualize(image)  # Draw the tracked boxes on the frame

----

.. note::

   This documentation provides detailed information about each model class, its methods, and usage examples. For further details on individual methods and arguments, refer to the respective code documentation.
