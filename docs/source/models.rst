Models Overview
===============

.. meta::
   :name: description
   :content: "The Kornia models overview provides detailed information about key built-in models for computer vision tasks, including real-time object detection (RT-DETR), edge detection (DexiNed), segmentation (UNet, DeepLabV3), and multi-object tracking (BoxMotTracker). It offers comprehensive documentation on each model, including methods, parameters, and example usage to streamline the integration of these models into computer vision workflows."


This section covers several of Kornia's built-in models for key computer vision tasks. Each model is documented with its respective API and example usage.

.. _ObjectDetectorBuilder:

ObjectDetectorBuilder
---------------------

The `ObjectDetectorBuilder` class is a builder for constructing a detection model based on the RT-DETR architecture, which is designed for real-time object detection. It is capable of detecting multiple objects within an image and provides efficient inference suitable for real-world applications.

**Key Methods:**

- `build`: Constructs and returns an instance of the RTDETR detection model.
- `save`: Saves the processed image or results after applying the detection model.

.. autoclass:: kornia.contrib.object_detection.ObjectDetectorBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code demonstrates how to use `ObjectDetectorBuilder` to detect objects in an image:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.contrib.object_detection.ObjectDetectorBuilder.build()
       model.save(image)

.. _EdgeDetectorBuilder:

EdgeDetectorBuilder
-------------------

The `EdgeDetectorBuilder` class implements a state-of-the-art edge detection model based on DexiNed, which excels at detecting fine-grained edges in images. This model is well-suited for tasks like medical imaging, object contour detection, and more.

**Key Methods:**

- `build`: Builds and returns an instance of the DexiNed edge detection model.
- `save`: Saves the detected edges for further processing or visualization.

.. autoclass:: kornia.contrib.edge_detection.EdgeDetectorBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code shows how to use the `EdgeDetectorBuilder` to detect edges in an image:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.contrib.edge_detection.EdgeDetectorBuilder.build()
       model.save(image)

.. _SegmentationModels:

SegmentationModelsBuilder
-------------------------

The `SegmentationModelsBuilder` class offers a flexible API for implementing and running various segmentation models. It supports a variety of architectures such as UNet, FPN, and others, making it highly adaptable for tasks like semantic segmentation, instance segmentation, and more.

**Key Methods:**

- `__init__`: Initializes a segmentation model based on the chosen architecture (e.g., UNet, DeepLabV3, etc.).
- `forward`: Runs inference on an input tensor and returns segmented output.

**Parameters:**

- `model_name`: (str) Name of the segmentation architecture to use, e.g., `"Unet"`, `"DeepLabV3"`.
- `classes`: (int) The number of output classes for segmentation.

.. autoclass:: kornia.models.segmentation.segmentation_models.SegmentationModelsBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   Here's an example of how to use `SegmentationModelsBuilder` for binary segmentation:

   .. code-block:: python

       import kornia
       input_tensor = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.models.segmentation.segmentation_models.SegmentationModelsBuilder.build()
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

.. autoclass:: kornia.contrib.boxmot_tracker.BoxMotTracker
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following example demonstrates how to track objects across multiple frames using `BoxMotTracker`:

   .. code-block:: python

       import kornia
       image = kornia.utils.sample.get_sample_images()[0][None]
       model = kornia.contrib.boxmot_tracker.BoxMotTracker()
       for i in range(4):
           model.update(image)  # Update the tracker with new frames
       model.save(image)       # Save the tracking result

---

.. note::

   This documentation provides detailed information about each model class, its methods, and usage examples. For further details on individual methods and arguments, refer to the respective code documentation.
