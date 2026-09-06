kornia.models
=============

.. meta::
   :description: The Kornia models overview provides detailed information about key built-in models for computer vision tasks, including real-time object detection (RT-DETR), edge detection (DexiNed) and semantic segmentation (a wrapper for segmentation_models_pytorch networks). It offers comprehensive documentation on each model, including methods, parameters, and example usage to streamline the integration of these models into computer vision workflows.


Builders for Kornia's ready-to-use models (object detection, edge detection, semantic segmentation and Kimi-VL).
Each builder returns a configured model with pretrained weights. For the papers behind the models, see the :doc:`Models </models/index>` section.
Pretrained weights are downloaded on first use, and the model builders return a regular ``nn.Module`` that accepts a
batched ``(B, 3, H, W)`` float image in ``[0, 1]``. :func:`kornia.io.get_sample_images` provides a couple of sample images for quick experiments.

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

.. _RRDBNet:

RRDBNet
-------

The `RRDBNet` class is the Residual-in-Residual Dense Block generator behind ESRGAN and Real-ESRGAN.
It is a plain ``nn.Module`` that upsamples a batched ``(B, 3, H, W)`` image by a factor of 1, 2 or 4,
and its module and parameter names match the reference implementation, so the published Real-ESRGAN
checkpoints load with ``strict=True``. ``kornia.contrib.super_resolution.RRDBNetBuilder`` configures
it for the released Real-ESRGAN variants and downloads their weights, but it currently raises
``TypeError`` because the ``SuperResolution`` wrapper it returns never implements the abstract
``from_config`` method it inherits (`kornia#4291 <https://github.com/kornia/kornia/issues/4291>`_);
until that is fixed, construct ``RRDBNet`` directly and load the checkpoint with ``load_state_dict``.

The architecture is vendored from `BasicSR <https://github.com/XPixelGroup/BasicSR>`_ (Apache-2.0,
Copyright 2018-2022 BasicSR Authors); no extra package is required to use it.

.. autoclass:: kornia.models.RRDBNet
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   The following code upsamples an image by a factor of 4 with a randomly initialized generator:

   .. code-block:: python

       import torch
       from kornia.models import RRDBNet

       model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23).eval()
       upsampled = model(torch.rand(1, 3, 32, 32))  # (1, 3, 128, 128)

.. _SegmentationModels:

SegmentationModelsBuilder
-------------------------

The `SegmentationModelsBuilder` class wraps a segmentation network you have already built -- typically one from
`segmentation_models_pytorch <https://github.com/qubvel-org/segmentation_models.pytorch>`_ (smp), but any
``nn.Module`` mapping ``(B, 3, H, W)`` to ``(B, C, H, W)`` works -- in a
:class:`~kornia.models.segmentation.SemanticSegmentation` container, prepending the ONNX-friendly preprocessing
(BGR-to-RGB, range rescaling, mean/std normalization) that the encoder's pretrained weights expect. Kornia does not
import smp; you build the network and fetch its preprocessing parameters yourself.

**Key Methods:**

- `build`: Wraps a constructed segmentation network and its encoder's preprocessing parameters.
- `get_preprocessing_pipeline`: Turns a preprocessing-parameter dictionary into an
  :class:`~kornia.augmentation.container.ImageSequential`.

**Main parameters of** `build`:

- `model`: (nn.Module) The segmentation network.
- `preproc_params`: (dict | None) The encoder's preprocessing parameters, in the shape returned by
  ``smp.encoders.get_preprocessing_params(encoder_name)``: ``input_space``, ``input_range``, ``mean`` and ``std``.
  ``None`` means the input is fed to the network unchanged.
- `name`: (str) The name of the wrapped model, used by ``save``.

.. autoclass:: kornia.models.segmentation.segmentation_models.SegmentationModelsBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   Here's an example of how to use `SegmentationModelsBuilder` with an smp UNet for two-class segmentation:

   .. code-block:: python

       import kornia
       import segmentation_models_pytorch as smp
       from kornia.models.segmentation import SegmentationModelsBuilder

       net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=2, activation="softmax2d")
       params = smp.encoders.get_preprocessing_params("resnet34")
       model = SegmentationModelsBuilder.build(net, params, name="Unet_resnet34")

       input_tensor = kornia.io.get_sample_images()[0][None]
       segmented_output = model(input_tensor)
       print(segmented_output.shape)  # (1, 2, H, W)

.. autoclass:: kornia.models.segmentation.SemanticSegmentation
   :members: forward, visualize, save

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

----

.. note::

   This documentation provides detailed information about each model class, its methods, and usage examples. For further details on individual methods and arguments, refer to the respective code documentation.
