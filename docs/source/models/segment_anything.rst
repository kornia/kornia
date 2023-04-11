Segment Anything (SAM)
======================

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it
can be used to generate masks for all objects in an image.

.. card::
    :link: https://segment-anything.com/

    **Segment Anything**
    ^^^
    **Abstract:** We introduce the Segment Anything (SA) project: a new task, model, and dataset for image
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

Load/build the model
^^^^^^^^^^^^^^^^^^^^
This is an example how you can load the model.

.. code-block:: python

    from kornia.contrib import Sam

    # model_type can be:
    #   0, 'vit_h' or `kornia.contrib.SamModelType.vit_h`
    #   1, 'vit_l' or `kornia.contrib.SamModelType.vit_l`
    #   2, 'vit_b' or `kornia.contrib.SamModelType.vit_b`
    model_type = 'vit_b' # or can be a number `2` or the enum kornia.contrib.SamModelType.vit_b

    # Load/build the model
    sam_model = Sam.build(model_type)

Load checkpoint
^^^^^^^^^^^^^^^
With the load checkpoint method you can load from a file or directly from a URL. The official (by meta) model weights are:

#. `vit_h`: `ViT-H SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth>`_.
#. `vit_l`: `ViT-L SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>`_.
#. `vit_b`: `ViT-B SAM model - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>`_.


.. code-block:: python

    from kornia.contrib import Sam
    from kornia.utils import get_cuda_device_if_available

    model_type = 'vit_b'

    checkpoint = './path_for_the_model_checkpoint.pth' # Can be a filepath or a url
    device = get_cuda_device_if_available()

    # Load/build the model
    sam_model = Sam.build(model_type)

    # Load the checkpoint
    sam_model.load_checkpoint(checkpoint, device)

Load from pretrained
^^^^^^^^^^^^^^^^^^^^
This method internally uses `build` and `load_checkpoint`, also move the model for the desired device.

.. code-block:: python

    from kornia.contrib import Sam
    from kornia.utils import get_cuda_device_if_available

    model_type = 'vit_b'

    checkpoint = './path_for_the_model_checkpoint.pth' # Can be a filepath or a url
    device = get_cuda_device_if_available()

    # Load the model with checkpoint on the desired device
    sam_model = Sam.from_pretrained(model_type, checkpoint, device)



Predictor
^^^^^^^^^
.. _anchor Predictor:

The High level API `SamPrediction` handle with the image and prompt transformation, preprocessing and prediction for
a given SAM model.

.. code-block:: python

    import torch

    from kornia.contrib import Sam
    from kornia.contrib.sam.predictor import SamPredictor
    from kornia.io import load_image, ImageLoadType
    from kornia.geometry.keypoints import Keypoints
    from kornia.geometry.boxes import Boxes
    from kornia.utils import get_cuda_device_if_available

    model_type = 'vit_h' # or can be a number `0` or the enum kornia.contrib.sam.SamModelType.vit_h
    checkpoint_path = './path_for_the_vit_h_checkpoint.pth'
    device = get_cuda_device_if_available()

    # Load the model
    sam_model = Sam.build(model_type, checkpoint_path, device)

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB8, device).float()

    # Transform the image (CxHxW) into a batched input (BxCxHxW)
    image = image[None, ...]

    # Load the predictor
    predictor = SamPredictor(sam_model)

    # Generate the prompts
    input_point = Keypoints(torch.tensor([[[500, 375]]], device=device, dtype=torch.float32)) # BxNx2
    input_label = torch.tensor([[1]], device=device) # BxN -- 1 indicates a foreground point; 0 indicates a background point
    input_box = Boxes(
        torch.tensor([[[[425, 600], [425, 875], [700, 600], [700, 875]]]], device=device, dtype=torch.float32), mode='xyxy'
    )

    # Runs the prediction with all prompts
    prediction = predictor(
        image=image,
        point_coords=input_point,
        point_labels=input_label,
        boxes=input_box,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction with just the keypoints
    prediction = predictor(
        image=image,
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction with just the box
    prediction = predictor(
        image=image,
        boxes=input_box,
        multimask_output=True,
    )

    #----------------------------------------------
    # or run the prediction without prompts
    prediction = predictor(
        image=image,
        multimask_output=True,
    )

    #------------------------------------------------
    # or run the prediction using the previous logits
    prediction = predictor(
        image=image,
        mask_input=prediction.logits
        multimask_output=True,
    )

    # The `prediction` is a dataclass with the masks, scores and logits
    print(prediction.masks.shape)
    print(prediction.scores)
    print(prediction.logits.shape)

.. Mask Generator
.. ^^^^^^^^^^^^^^


Example of how to use the SAM model without API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a simple example, of how to directly use the SAM model loaded. We recommend the use of
:ref:`Predictor API<anchor Predictor>` to handle/prepare the inputs.

.. code-block:: python

    from kornia.contrib import Sam
    from kornia.io import load_image, ImageLoadType
    from kornia.utils import get_cuda_device_if_available
    from kornia.geometry import resize

    model_type = 'vit_b' # or can be a number `2` or the enum sam.SamModelType.vit_b
    checkpoint_path = './path_for_the_model_checkpoint.pth'
    device = get_cuda_device_if_available()

    # Load the model
    sam_model = Sam.from_pretrained(model_type, checkpoint_path, device)

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB8, device).float()

    # Transform the image (CxHxW) into a batched input (BxCxHxW)
    image = image[None, ...]

    # Resize the image to have the maximum size 1024 on its largest side
    inpt = resize(image, 1024, side='long')

    # Embed prompts -- ATTENTION: should match the coordinates after the resize of the image
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

    # Preprocess input
    input_image = sam_model.preprocess(inpt)

    # Predict masks
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=sam_model.image_encoder(input_image),
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )

    # Upscale the masks to the original image resolution
    input_shape = (inpt.shape[-2], inpt.shape[-1])
    original_shape = (image.shape[-2], image.shape[-1])
    masks = sam_model.postprocess_masks(low_res_masks, input_shape, original_shape)

    # If wants to have a binary mask
    masks = masks > sam_model.mask_threshold

    # To transform it into a SamPrediction
    sam_preds = sam.model.SamPrediction(masks, iou_predictions, low_res_masks)
