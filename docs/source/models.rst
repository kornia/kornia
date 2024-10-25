Models Overview
===============

This section covers several of Kornia's built-in models for key computer vision tasks. Each model is documented with its respective API and example usage.

.. _RTDETRDetectorBuilder:

RTDETRDetectorBuilder
---------------------

The `RTDETRDetectorBuilder` class is a builder for constructing a detection model based on the RT-DETR architecture, which is designed for real-time object detection. It is capable of detecting multiple objects within an image and provides efficient inference suitable for real-world applications.

**Key Methods:**

- `build`: Constructs and returns an instance of the RTDETR detection model.
- `save`: Saves the processed image or results after applying the detection model.

.. autoclass:: kornia.models.detection.rtdetr.RTDETRDetectorBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code demonstrates how to use `RTDETRDetectorBuilder` to detect objects in an image:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.models.detection.rtdetr.RTDETRDetectorBuilder.build()
       model.save(image)

.. _DexiNedBuilder:

DexiNedBuilder
--------------

The `DexiNedBuilder` class implements a state-of-the-art edge detection model based on DexiNed, which excels at detecting fine-grained edges in images. This model is well-suited for tasks like medical imaging, object contour detection, and more.

**Key Methods:**

- `build`: Builds and returns an instance of the DexiNed edge detection model.
- `save`: Saves the detected edges for further processing or visualization.

.. autoclass:: kornia.models.edge_detection.dexined.DexiNedBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code shows how to use the `DexiNedBuilder` to detect edges in an image:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.models.edge_detection.dexined.DexiNedBuilder.build()
       model.save(image)

.. _SegmentationModels:

SegmentationModels
------------------

The `SegmentationModels` class offers a flexible API for implementing and running various segmentation models. It supports a variety of architectures such as UNet, FPN, and others, making it highly adaptable for tasks like semantic segmentation, instance segmentation, and more.

**Key Methods:**

- `__init__`: Initializes a segmentation model based on the chosen architecture (e.g., UNet, DeepLabV3, etc.).
- `forward`: Runs inference on an input tensor and returns segmented output.

**Parameters:**

- `model_name`: (str) Name of the segmentation architecture to use, e.g., `"Unet"`, `"DeepLabV3"`.
- `classes`: (int) The number of output classes for segmentation.

.. autoclass:: kornia.models.segmentation.segmentation_models.SegmentationModels
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   Here's an example of how to use `SegmentationModels` for binary segmentation:

   .. code-block:: python

       import kornia
       input_tensor = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.models.segmentation.segmentation_models.SegmentationModels()
       segmented_output = model(input_tensor)
       print(segmented_output.shape)

.. _BoxMotTracker:

BoxMotTracker
-------------

The `BoxMotTracker` class is used for multi-object tracking in video streams. It is designed to track bounding boxes of objects across multiple frames, supporting various tracking algorithms for object detection and tracking continuity.

**Key Methods:**

- `__init__`: Initializes the multi-object tracker.
- `update`: Updates the tracker with a new image frame.
- `save`: Saves the tracked object data or visualization for post-processing.

**Parameters:**

- `max_lost`: (int) The maximum number of frames where an object can be lost before it is removed from the tracker.

.. autoclass:: kornia.models.tracking.boxmot_tracker.BoxMotTracker
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following example demonstrates how to track objects across multiple frames using `BoxMotTracker`:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.models.tracking.boxmot_tracker.BoxMotTracker()
       for i in range(4):
           model.update(image)  # Update the tracker with new frames
       model.save(image)       # Save the tracking result

---

.. note::

   This documentation provides detailed information about each model class, its methods, and usage examples. For further details on individual methods and arguments, refer to the respective code documentation.
