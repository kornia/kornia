from __future__ import annotations

from enum import Enum

import torch

from kornia.contrib.sam.architecture import Sam as SAM_ARCH
from kornia.contrib.sam.architecture.common import LayerNorm
from kornia.contrib.sam.architecture.image_encoder import ImageEncoderViT
from kornia.contrib.sam.architecture.mask_decoder import MaskDecoder
from kornia.contrib.sam.architecture.prompt_encoder import PromptEncoder
from kornia.contrib.sam.architecture.transformer import TwoWayTransformer
from kornia.contrib.sam.base import ModelBase


class SamModelType(Enum):
    """Map the SAM model types."""

    vit_h = 0
    vit_l = 1
    vit_b = 2


class Sam(ModelBase, SAM_ARCH):
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
