Segment Anything (SAM)
======================

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it
can be used to generate masks for all objects in an image.

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



How to use SAM from Kornia
--------------------------
The Kornia API for SAM try to provide a simple API to access initialize the model and load/download the weights. Also,
providing it to a high-level API called :code:`VisualPrompter`, which allow the users to set an image and run multiple
queries multiple times.

The :code:`VisualPrompter` works querying on a single image, if you want to explore and query into a batch of images,
you can use the :code:`Sam` directly. But, for it you will need to write the boilerplate to preprocess and postprocess to
use it. This boilerplate, is already handle on the high-level API :code:`VisualPrompter`.

Image Prompter
^^^^^^^^^^^^^^
.. _anchor Prompter:

The High level API :code:`VisualPrompter` handle with the image and prompt transformation, preprocessing and prediction for
a given SAM model.

About the :code:`VisualPrompter`:

#. From a `ModelConfig` loads the desired model with the desired checkpoint to be used as the model to receive the query
   prompts. For know we just support Segment Anything model, where the *SAM-h* is the default option.

#. Based on the model, the :code:`VisualPrompter` will handle with the necessary transformations to be done into the image
   and prompts before apply it to the model. These transformations are done using PyTorch backed, by our API of
   augmentations. Where we use the :class:`kornia.geometry.augmentation.AugmentationSequential` to handle with the different
   data formats (keypoints, boxes, masks, image).

#. When you use :code:`prompter.set_image(...)`, the prompter will preprocess this image, and then pass it to the encoder,
   and cache the embeddings to query it after.

    * The preprocess steps are: 1) Resize the image to have its longer side the same size as :code:`image_encoder` image size
      input. 2) Cache the information of this transformation to apply into the prompts. 3) normalize the image based on the
      passed mean and standard deviation, or with the values of the SAM dataset. 4) pad on the bottom and right for the image
      have the encoder expected resolution: :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`.

    * The best image to be used will always have the shape equals to
      :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`.

#. When you use :code:`prompter.predict(...)`, the prompter will apply the cached transformations on the coordinates of the
   prompts, and then query this prompts into the cached embeddings.

    * If :code:`output_original_size=True`, the results structure will upsample the logits from it's resolution into the
      image input original resolution. The output logits has the height and width equals to 256.

#. You can benefit from using the :code:`torch.compile(...)` API (dynamo) for torch >= 2.0.0 version. To compile with dynamo
   we provide the method :code:`prompter.compile(...)` which will optimize the right parts of the backend model and the
   prompter itself.

--------------

Example of using the :code:`VisualPrompter`:

Exploring how to simple initialize the :code:`VisualPrompter`, automatically load the weights from a URL,
read the image and set it to be query, how to write the prompts, and the multiple ways we can use these prompts
to query the image masks from the SAM model.


.. code-block:: python

    import torch

    from kornia.contrib.models.sam import SamConfig
    from kornia.contrib.visual_prompter import VisualPrompter
    from kornia.io import load_image, ImageLoadType
    from kornia.geometry.keypoints import Keypoints
    from kornia.geometry.boxes import Boxes
    from kornia.utils import get_cuda_or_mps_device_if_available

    model_type = 'vit_h'
    checkpoint = './https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    device = get_cuda_or_mps_device_if_available

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
    boxes = Boxes(
        torch.tensor([[[[425, 600], [425, 875], [700, 600], [700, 875]]]], device=device, dtype=torch.float32), mode='xyxy'
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
    # or run the prediction using the previous logits
    prediction = prompter.predict(
        masks=prediction.logits
        multimask_output=True,
    )

    # The `prediction` is a SegmentationResults dataclass with the masks, scores and logits
    print(prediction.masks.shape)
    print(prediction.scores)
    print(prediction.logits.shape)


Read more about the :code:`SegmentationResults` on :ref:`the official docs<anchor SegmentationResults>`



Load from config
^^^^^^^^^^^^^^^^
You can build a SAM model by specifying the encoder parameters on the the :code:`SamConfig`, or from the model type. The
:code:`from_config` method will first try to build the model based on the model type, otherwise will try from the specified
parameters. If a checkpoint URL or path for a file is seted, the method will automatically load it.

.. code-block:: python

    from kornia.contrib.models.sam import Sam, SamConfig
    from kornia.utils import get_cuda_or_mps_device_if_available

    # model_type can be:
    #   0, 'vit_h' or `kornia.contrib.models.sam.SamModelType.vit_h`
    #   1, 'vit_l' or `kornia.contrib.models.sam.SamModelType.vit_l`
    #   2, 'vit_b' or `kornia.contrib.models.sam.SamModelType.vit_b`
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
With the load checkpoint method you can load from a file or directly from a URL. The official (by meta) model weights are:

#. `vit_h`: `ViT-H SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_.
#. `vit_l`: `ViT-L SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`_.
#. `vit_b`: `ViT-B SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`_.

If a URL is passed the model will automatically download and cache the weights using
:code:`torch.hub.load_state_dict_from_url`

.. code-block:: python

    from kornia.contrib.models.sam import Sam, SamConfig
    from kornia.utils import get_cuda_or_mps_device_if_available

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


Example of how to use the SAM model without API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a simple example, of how to directly use the SAM model loaded. We recommend the use of
:ref:`Prompter API<anchor Prompter>` to handle/prepare the inputs.

.. code-block:: python

    from kornia.contrib.models.sam import Sam
    from kornia.contrib.models import SegmentationResults
    from kornia.io import load_image, ImageLoadType
    from kornia.utils import get_cuda_or_mps_device_if_available
    from kornia.geometry import resize
    from kornia.enhance import normalize

    model_type = 'vit_b' # or can be a number `2` or the enum sam.SamModelType.vit_b
    checkpoint_path = './path_for_the_model_checkpoint.pth'
    device = get_cuda_or_mps_device_if_available()

    # Load the model
    sam_model = Sam.from_pretrained(model_type, checkpoint_path, device)

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB32, device)

    # Transform the image (CxHxW) into a batched input (BxCxHxW)
    image = image[None, ...]

    # Resize the image to have the maximum size 1024 on its largest side
    inpt = resize(image, 1024, side='long')

    # Embed prompts -- ATTENTION: should match the coordinates after the resize of the image
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

    # define the info for normalize the input
    pixel_mean = torch.tensor(...)
    pixel_std = torch.tensor(...)

    # Preprocess input
    inpt = normalize(inpt, pixel_mean, pixel_std)
    padh = model_sam.image_encoder.img_size - h
    padw = model_sam.image_encoder.img_size - w
    inpt = pad(inpt, (0, padw, 0, padh))

    #--------------------------------------------------------------------
    # Option A: Manually calling each API
    #--------------------------------------------------------------------
    low_res_logits, iou_predictions = sam_model.mask_decoder(
        image_embeddings=sam_model.image_encoder(inpt),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )

    prediction = SegmentationResults(low_res_logits, iou_predictions)

    #--------------------------------------------------------------------
    # Option B: Calling the model itself
    #--------------------------------------------------------------------
    prediction = sam_model(inpt[None, ...], [{}], multimask_output=True)

    #--------------------------------------------------------------------
    # Post processing
    #--------------------------------------------------------------------
    # Upscale the masks to the original image resolution
    input_size = (inpt.shape[-2], inpt.shape[-1])
    original_size = (image.shape[-2], image.shape[-1])
    image_size_encoder = (model_sam.image_encoder.img_size, model_sam.image_encoder.img_size)
    prediction.original_res_logits(input_size, original_size, image_size_encoder)

    # If wants to check the binary masks
    masks = prediction.binary_masks
