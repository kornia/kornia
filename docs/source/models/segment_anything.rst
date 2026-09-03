Segment Anything (SAM)
======================

.. rst-class:: kornia-badges

:bdg-primary:`Segmentation` :bdg-primary:`Visual prompting` :bdg-secondary:`Apache-2.0`

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes.
Kornia ships the model as :class:`~kornia.models.sam.Sam` and a high-level
:class:`~kornia.contrib.visual_prompter.VisualPrompter` that encodes an image once and answers any number of prompt
queries against it, returning a :class:`~kornia.models.structures.SegmentationResults` with the mask logits, the
predicted IoU scores and thresholded ``binary_masks``.

Run it
------

.. code-block:: python

    import torch
    from kornia.io import load_image
    from kornia.models.sam import SamConfig
    from kornia.contrib.visual_prompter import VisualPrompter
    from kornia.geometry.boxes import Boxes
    from kornia.geometry.keypoints import Keypoints

    image = load_image("simba.png")  # (3, H, W) float in [0, 1]

    prompter = VisualPrompter(SamConfig("vit_b", pretrained=True))  # vit_h, vit_l, vit_b or mobile_sam
    prompter.set_image(image)  # encode once, query many times

    # a foreground point (label 1; 0 would be background) -> three candidate masks
    keypoints = Keypoints(torch.tensor([[[300.0, 90.0]]]))  # (K, N, 2): K prompts of N (x, y) points, in pixels
    prediction = prompter.predict(keypoints=keypoints, keypoints_labels=torch.tensor([[1]]))
    best = prediction.binary_masks[0, prediction.scores.argmax()]  # (H, W) bool, highest predicted IoU

    # a box prompt -> a single mask
    box = Boxes.from_tensor(torch.tensor([[[180.0, 20.0, 380.0, 240.0]]]), mode="xyxy")
    prediction = prompter.predict(boxes=box, multimask_output=False)
    mask = prediction.binary_masks[0, 0]  # (H, W) bool

.. figure:: /_static/img/models/segment_anything.jpg
   :align: center
   :alt: A cartoon lion cub with a point and a box prompt drawn on it (left), and the SAM masks for the point prompt and the box prompt overlaid in green (centre, right).

   ``vit_b`` masks for the point prompt (best of the three candidates) and the box prompt, with the predicted IoU
   scores in the titles.

The sections below cover the full prompter API, building :class:`~kornia.models.sam.Sam` from a config or a
checkpoint, and calling the model without the prompter.

How to use SAM from Kornia
--------------------------
The Kornia API for SAM provides a simple way to initialize the model and load or download its weights, together with a
high-level API called :code:`VisualPrompter`, which allows users to set an image once and run multiple queries against it.

The :code:`VisualPrompter` works on a single image. If you want to query a batch of images, you can use
:code:`Sam` directly, but then you need to write the pre- and post-processing boilerplate yourself. That boilerplate is
already handled by the high-level :code:`VisualPrompter` API.

Visual Prompter
^^^^^^^^^^^^^^^
.. _anchor Prompter:

The high-level :code:`VisualPrompter` API handles the image and prompt transformations, preprocessing and prediction for
a given SAM model.

About the :code:`VisualPrompter`:

#. From a :class:`~kornia.models.sam.SamConfig` it loads the desired model with the desired checkpoint, which then receives
   the query prompts. For now only the Segment Anything family is supported, with *SAM ViT-H* as the default option.

#. Based on the model, the :code:`VisualPrompter` applies the necessary transformations to the image and prompts before
   passing them to the model. These transformations are implemented with our augmentation API, using
   :class:`kornia.augmentation.AugmentationSequential` to handle the different data formats (keypoints, boxes, masks, image).

#. When you call :code:`prompter.set_image(...)`, the prompter preprocesses the image, passes it to the encoder,
   and caches the embeddings for later queries. Note that the image should be scaled to the range [0, 1].

    * The preprocessing steps are: 1) resize the image so that its longer side matches the :code:`image_encoder` input size;
      2) cache this transformation so it can be applied to the prompts; 3) normalize the image with the given mean and
      standard deviation, or with the SAM dataset statistics; 4) pad the bottom and right so the image has the resolution the
      encoder expects: :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`.

    * The best input image will always have shape
      :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`.

#. When you call :code:`prompter.predict(...)`, the prompter applies the cached transformation to the prompt coordinates
   and then queries the cached embeddings with them.

    * If :code:`output_original_size=True`, the result structure upsamples the logits from their native resolution to the
      original resolution of the input image. The raw logits have a height and width of 256.

#. You can benefit from the :code:`torch.compile(...)` API (dynamo). To compile with dynamo
   we provide the :code:`prompter.compile(...)` method, which optimizes the right parts of the backend model and of the
   prompter itself.

--------------

Example of using the :code:`VisualPrompter`:

The example shows how to initialize the :code:`VisualPrompter`, automatically load the weights from a URL,
read an image and set it as the query target, how to write the prompts, and the multiple ways these prompts
can be used to query the SAM model for image masks.


.. code-block:: python

    import torch

    from kornia.models.sam import SamConfig
    from kornia.contrib.visual_prompter import VisualPrompter
    from kornia.io import load_image, ImageLoadType
    from kornia.geometry.keypoints import Keypoints
    from kornia.geometry.boxes import Boxes
    from kornia.core.utils import get_cuda_or_mps_device_if_available

    model_type = 'vit_h'
    checkpoint = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    device = get_cuda_or_mps_device_if_available()

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB32, device)

    # Define the model config
    config = SamConfig(model_type, checkpoint)

    # Load the prompter
    prompter = VisualPrompter(config, device=device)

    # You can use torch dynamo/compile API with:
    # prompter.compile()

    # set the image: This will preprocess the image and already generate the embeddings of it
    prompter.set_image(image)

    # Generate the prompts
    keypoints = Keypoints(torch.tensor([[[500, 375]]], device=device, dtype=torch.float32)) # BxNx2
    # For the keypoints label: 1 indicates a foreground point; 0 indicates a background point
    keypoints_labels = torch.tensor([[1]], device=device) # BxN
    boxes = Boxes.from_tensor(
        torch.tensor([[[425, 600, 700, 875]]], device=device, dtype=torch.float32), mode='xyxy'  # BxNx4
    )

    # Runs the prediction with all prompts
    prediction = prompter.predict(
        keypoints=keypoints,
        keypoints_labels=keypoints_labels,
        boxes=boxes,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction with just the keypoints
    prediction = prompter.predict(
        keypoints=keypoints,
        keypoints_labels=keypoints_labels,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction with just the box
    prediction = prompter.predict(
        boxes=boxes,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction without prompts
    prediction = prompter.predict(
        multimask_output=True,
    )

    #------------------------------------------------
    # or run the prediction using the previous logits as a mask prompt: one (K, 1, 256, 256)
    # low-resolution mask per query, so pick the candidate with the best predicted IoU
    best = prediction.scores[0].argmax()
    prediction = prompter.predict(
        masks=prediction.logits[:, best : best + 1],
        multimask_output=True,
    )

    # The `prediction` is a SegmentationResults dataclass with the logits, scores and thresholded binary masks
    print(prediction.binary_masks.shape)
    print(prediction.scores)
    print(prediction.logits.shape)


Read more about :code:`SegmentationResults` in :ref:`the API reference <anchor SegmentationResults>`.



Load from config
^^^^^^^^^^^^^^^^
You can build a SAM model by specifying the encoder parameters in the :code:`SamConfig`, or just the model type. The
:code:`from_config` method first tries to build the model from the model type, and otherwise falls back to the specified
parameters. If a checkpoint URL or file path is set, the method loads it automatically.

.. code-block:: python

    from kornia.models.sam import Sam, SamConfig
    from kornia.core.utils import get_cuda_or_mps_device_if_available

    # model_type can be:
    #   0, 'vit_h' or `kornia.models.sam.SamModelType.vit_h`
    #   1, 'vit_l' or `kornia.models.sam.SamModelType.vit_l`
    #   2, 'vit_b' or `kornia.models.sam.SamModelType.vit_b`
    model_type = 'vit_b'

    # The checkpoint can be a filepath or a url
    checkpoint = './path_for_the_model_checkpoint.pth'
    device = get_cuda_or_mps_device_if_available()

    # Load config
    config = SamConfig(model_type, checkpoint)

    # Load the model with checkpoint
    sam_model = Sam.from_config(config)

    # Move to desired device
    sam_model = sam_model.to(device)


Load checkpoint
^^^^^^^^^^^^^^^
With the :code:`load_checkpoint` method you can load weights from a file or directly from a URL. The official model weights released by Meta are:

#. `vit_h`: `ViT-H SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_.
#. `vit_l`: `ViT-L SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`_.
#. `vit_b`: `ViT-B SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`_.

If a URL is passed, the model automatically downloads and caches the weights using
:code:`torch.hub.load_state_dict_from_url`.

.. code-block:: python

    from kornia.models.sam import Sam, SamConfig
    from kornia.core.utils import get_cuda_or_mps_device_if_available

    model_type = 'vit_b'

    # The checkpoint can be a filepath or a url
    checkpoint = './path_for_the_model_checkpoint.pth'
    device = get_cuda_or_mps_device_if_available()

    # Load/build the model
    sam_model = Sam.from_config(SamConfig(model_type))

    # Load the checkpoint
    sam_model.load_checkpoint(checkpoint, device)


.. Mask Generator
.. ^^^^^^^^^^^^^^


Example of how to use the SAM model without the prompter API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a simple example of how to use the loaded SAM model directly. We recommend using the
:ref:`prompter API <anchor Prompter>` to handle and prepare the inputs.

.. code-block:: python

    import torch
    from torch.nn.functional import pad

    from kornia.models.sam import Sam, SamConfig
    from kornia.models.structures import SegmentationResults
    from kornia.io import load_image, ImageLoadType
    from kornia.core.utils import get_cuda_or_mps_device_if_available
    from kornia.geometry import resize
    from kornia.enhance import normalize

    model_type = 'vit_b'  # or the number `2`, or the enum `SamModelType.vit_b`
    checkpoint_path = './path_for_the_model_checkpoint.pth'
    device = get_cuda_or_mps_device_if_available()

    # Load the model
    sam_model = Sam.from_config(SamConfig(model_type, checkpoint_path)).to(device)

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB32, device)

    # Transform the image (CxHxW) into a batched input (BxCxHxW)
    image = image[None, ...]

    # Resize the image to have a maximum size of 1024 on its largest side
    data = resize(image, 1024, side='long')
    h, w = data.shape[-2:]

    # Embed prompts -- ATTENTION: prompt coordinates must match the image after the resize
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

    # SAM dataset statistics used to normalize the input, in the [0, 1] range
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device) / 255.0
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device) / 255.0

    # Preprocess input
    data = normalize(data, pixel_mean, pixel_std)
    padh = sam_model.image_encoder.img_size - h
    padw = sam_model.image_encoder.img_size - w
    data = pad(data, (0, padw, 0, padh))

    #--------------------------------------------------------------------
    # Option A: Manually calling each API
    #--------------------------------------------------------------------
    low_res_logits, iou_predictions = sam_model.mask_decoder(
        image_embeddings=sam_model.image_encoder(data),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )

    prediction = SegmentationResults(low_res_logits, iou_predictions)

    #--------------------------------------------------------------------
    # Option B: Calling the model itself
    #--------------------------------------------------------------------
    prediction = sam_model(data, [{}], multimask_output=True)[0]  # one SegmentationResults per image

    #--------------------------------------------------------------------
    # Post processing
    #--------------------------------------------------------------------
    # Upscale the masks to the original image resolution
    input_size = (data.shape[-2], data.shape[-1])
    original_size = (image.shape[-2], image.shape[-1])
    image_size_encoder = (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size)
    prediction.original_res_logits(input_size, original_size, image_size_encoder)

    # Binary masks, thresholded from the logits
    masks = prediction.binary_masks


Paper
-----

.. card::
    :link: https://segment-anything.com/

    **Segment Anything**
    ^^^
    **Abstract:** We introduce the Segment Anything (SAM) project: a new task, model, and dataset for image
    segmentation. Using our efficient model in a data collection loop, we built the largest segmentation
    dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The
    model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions
    and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive
    -- often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything
    Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster
    research into foundation models for computer vision.

    **Tasks:** Segmentation

    **Datasets:** SA-1B

    **Licence:** Apache

    +++
    **Authors:** Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura
    Gustafson and Alex Berg and Wan-Yen Lo and Piotr Dollar and Ross Girshick
