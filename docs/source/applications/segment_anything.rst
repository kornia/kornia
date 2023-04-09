Segment Anything (SAM)
======================

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it
can be used to generate masks for all objects in an image.

How to use SAM from Kornia
--------------------------

Load/build the model
^^^^^^^^^^^^^^^^^^^^
This is an example how you can load the model.

.. code-block:: python

    from kornia.contrib import sam
    from kornia.utils import get_cuda_device_if_available

    # model_type can be:
    #   0, 'vit_h' or `kornia.contrib.sam.SamType.vit_h`
    #   1, 'vit_l' or `kornia.contrib.sam.SamType.vit_l`
    #   2, 'vit_b' or `kornia.contrib.sam.SamType.vit_b`
    model_type = 'vit_b' # or can be a number `2` or the enum sam.SamType.vit_b

    checkpoint_path = './path_for_the_model_checkpoint.pth'
    device = get_cuda_device_if_available()

    # Load the model
    sam_model = sam.load(model_type, checkpoint_path, device)

Predictor
^^^^^^^^^
.. _anchor Predictor:

The High level API `SamPrediction` handle with the image and prompt transformation, preprocessing and prediction for
a given SAM model.

.. code-block:: python

    import torch

    from kornia.contrib import sam
    from kornia.io import load_image, ImageLoadType
    from kornia.geometry.keypoints import Keypoints
    from kornia.geometry.boxes import Boxes
    from kornia.utils import get_cuda_device_if_available

    model_type = 'vit_h' # or can be a number `0` or the enum sam.SamType.vit_h
    checkpoint_path = './path_for_the_vit_h_checkpoint.pth'
    device = get_cuda_device_if_available()

    # Load the model
    sam_model = sam.load(model_type, checkpoint_path, device)

    # Load image
    image = load_image('./example.jpg', ImageLoadType.RGB8, device).float()

    # Transform the image (CxHxW) into a batched input (BxCxHxW)
    image = image[None, ...]

    # Load the predictor
    predictor = sam.SamPredictor(sam_model)

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

    from kornia.contrib import sam
    from kornia.io import load_image, ImageLoadType
    from kornia.utils import get_cuda_device_if_available
    from kornia.geometry import resize

    # model_type can be:
    #   0, 'vit_h' or `kornia.contrib.sam.SamType.vit_h`
    #   1, 'vit_l' or `kornia.contrib.sam.SamType.vit_l`
    #   2, 'vit_b' or `kornia.contrib.sam.SamType.vit_b`
    model_type = 'vit_b' # or can be a number `2` or the enum sam.SamType.vit_b
    checkpoint_path = './path_for_the_model_checkpoint.pth'
    device = get_cuda_device_if_available()

    # Load the model
    sam_model = sam.load(model_type, checkpoint_path, device)

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




Reference
---------
.. code-block:: latex

    @article{kirillov2023segany,
        title={Segment Anything},
        author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson,
        Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and
        Girshick, Ross},
        journal={arXiv:2304.02643},
        year={2023}
    }

Original implementation: <https://github.com/facebookresearch/segment-anything>
