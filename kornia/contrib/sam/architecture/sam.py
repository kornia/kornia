from __future__ import annotations

import os
from enum import Enum
from typing import Any

import torch
from torch.nn import functional as F

from kornia.contrib.sam.architecture.common import LayerNorm
from kornia.contrib.sam.architecture.image_encoder import ImageEncoderViT
from kornia.contrib.sam.architecture.mask_decoder import MaskDecoder
from kornia.contrib.sam.architecture.prompt_encoder import PromptEncoder
from kornia.contrib.sam.architecture.transformer import TwoWayTransformer
from kornia.core import Module, Tensor, pad, stack, tensor


class SamModelType(Enum):
    """Map the SAM model types."""

    vit_h = 0
    vit_l = 1
    vit_b = 2


class Sam(Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder: The backbone used to encode the image into image embeddings that allow for efficient mask
            prediction.
            prompt_encoder: Encodes various types of input prompts.
            mask_decoder: Predicts masks from the image embeddings and encoded prompts.
            pixel_mean: Mean values for normalizing pixels in the input image.
            pixel_std: Std values for normalizing pixels in the input image.
            device: The desired device to be used in the model
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", tensor(pixel_std).view(-1, 1, 1), False)

    def preprocess(self, x: Tensor) -> Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = pad(x, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def forward(self, batched_input: list[dict[str, Any]], multimask_output: bool) -> list[dict[str, Tensor]]:
        """Predicts masks end-to-end from provided images and prompts.

        If prompts are not known in advance, using SamPredictor is recommended over calling the model directly.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys. A prompt key can be
                excluded if it is not present. The options are:

                - "image": The image as a torch tensor in 3xHxW format, already transformed for input to the model.
                - "original_size": (tuple(int, int)) The original size of the image before transformation, as (H, W).
                - "point_coords": (Tensor) Batched point prompts for this image, with shape BxNx2. Already
                  transformed to the input frame of the model.
                - "point_labels": (Tensor) Batched labels for point prompts, with shape BxN.
                - "boxes": (Tensor) Batched box inputs, with shape Bx4. Already transformed to the input frame of the
                  model.
                - "mask_inputs": (Tensor) Batched mask inputs to the model, in the form Bx1xHxW.

            multimask_output: Whether the model should predict multiple disambiguating masks, or return a single mask.

        Returns:
            A list over input images, where each element is as dictionary with the following keys.
                - "masks": (Tensor) Batched binary mask predictions, with shape BxCxHxW, where B is the number
                  of input prompts, C is determiend by multimask_output, and (H, W) is the original size of the image.
                - "iou_predictions": (Tensor) The model's predictions of mask quality, in shape BxC.
                - "low_res_logits": (Tensor) Low resolution logits with shape BxCxHxW, where H=W=256. Can be passed as
                  mask input to subsequent iterations of prediction.
        """
        input_images = stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points, boxes=image_record.get("boxes", None), masks=image_record.get("mask_inputs", None)
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding[None, ...],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks, input_size=image_record["image"].shape[-2:], original_size=image_record["original_size"]
            )
            masks = masks > self.mask_threshold
            outputs.append({"masks": masks, "iou_predictions": iou_predictions, "low_res_logits": low_res_masks})
        return outputs

    def postprocess_masks(self, masks: Tensor, input_size: tuple[int, int], original_size: tuple[int, int]) -> Tensor:
        """Remove padding and upscale masks to the original image size.

        Args:
            masks: Batched masks from the mask_decoder, in BxCxHxW format.
            input_size: The size of the image input to the model, in (H, W) format. Used to remove padding.
            original_size: The original size of the image before resizing for input to the model, in (H, W) format.

        Returns:
            Batched masks in BxCxHxW format, where (H, W) is given by original_size.
        """
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def load_checkpoint(self, checkpoint: str, device: torch.device | None = None) -> None:
        """Load checkpoint from a given url or file.

        Args:
            checkpoint: The url or filepath for the respective checkpoint
            device: The desired device to load the weights and move the model
        """

        if os.path.isfile(checkpoint):
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location=device)

        self.load_state_dict(state_dict)

    @staticmethod
    def build(model_type: str | int | SamModelType) -> Sam:
        """Build the SAM model based on it's type.

        Args:
           model_type: the available models are:

               - 0, 'vit_h' or :func:`kornia.contrib.sam.SamModelType.vit_h`
               - 1, 'vit_l' or :func:`kornia.contrib.sam.SamModelType.vit_l`
               - 2, 'vit_b' or :func:`kornia.contrib.sam.SamModelType.vit_b`
        Returns:
            The respective SAM model

        Example:
            >>> sam_model = Sam.build('vit_b')
        """

        if isinstance(model_type, int):
            model_type = SamModelType(model_type)
        elif isinstance(model_type, str):
            _map_sam_type = {'vit_h': SamModelType.vit_h, 'vit_l': SamModelType.vit_l, 'vit_b': SamModelType.vit_b}
            model_type = _map_sam_type[model_type]

        if model_type == SamModelType.vit_b:
            model = _build_sam(
                encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, encoder_global_attn_indexes=(2, 5, 8, 11)
            )

        elif model_type == SamModelType.vit_l:
            model = _build_sam(
                encoder_embed_dim=1024,
                encoder_depth=24,
                encoder_num_heads=16,
                encoder_global_attn_indexes=(5, 11, 17, 23),
            )

        elif model_type == SamModelType.vit_h:
            model = _build_sam(
                encoder_embed_dim=1280,
                encoder_depth=32,
                encoder_num_heads=16,
                encoder_global_attn_indexes=(7, 15, 23, 31),
            )
        else:
            raise NotImplementedError('Unexpected model type. Should be vit_b, vit_l or vit_h.')

        return model

    @staticmethod
    def from_pretrained(
        model_type: str | int | SamModelType, checkpoint: str | None = None, device: torch.device | None = None
    ) -> Sam:
        """This builds the desired SAM model, load the checkpoint and move the model into the desired device.

        Args:
            model_type: the available models are:

                - 0, 'vit_h' or :func:`kornia.contrib.sam.SamModelType.vit_h`
                - 1, 'vit_l' or :func:`kornia.contrib.sam.SamModelType.vit_l`
                - 2, 'vit_b' or :func:`kornia.contrib.sam.SamModelType.vit_b`

            checkpoint: The url or filepath for the respective checkpoint
            device: The desired device to load the weights and move the model

        Returns:
            The respective SAM model

        Example:
            >>> # Input should be a RGB batched image
            >>> inpt = torch.randint(0, 255, (1, 3, 384, 384)).float()
            >>> inpt_after_resize = kornia.geometry.resize(inpt, (256, 256))
            >>> sam_model = Sam.from_pretrained('vit_b', 'cpu', checkpoint=None)
            >>> # Embed prompts
            >>> sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)
            >>> # Preprocess input (expected max size to be 1024)
            >>> input_image = sam_model.preprocess(inpt_after_resize)
            >>> # Predict masks
            >>> low_res_masks, iou_predictions = sam_model.mask_decoder(
            ...    image_embeddings=sam_model.image_encoder(input_image),
            ...    image_pe=sam_model.prompt_encoder.get_dense_pe(),
            ...    sparse_prompt_embeddings=sparse_embeddings,
            ...    dense_prompt_embeddings=dense_embeddings,
            ...    multimask_output=True,
            ... )
            >>> # Upscale the masks to the original image resolution
            >>> input_shape = (inpt_after_resize.shape[-2], inpt_after_resize.shape[-1])
            >>> original_shape = (inpt.shape[-2], inpt.shape[-1])
            >>> masks = sam_model.postprocess_masks(low_res_masks, input_shape, original_shape)
            >>> # If wants to have a binary mask
            >>> masks = masks > sam_model.mask_threshold
        """

        model = Sam.build(model_type)

        if checkpoint:
            model.load_checkpoint(checkpoint, device)

        if device:
            model = model.to(device=device)

        return model


def _build_sam(
    encoder_embed_dim: int, encoder_depth: int, encoder_num_heads: int, encoder_global_attn_indexes: tuple[int, ...]
) -> Sam:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    return Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=LayerNorm,
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
