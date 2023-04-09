from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from kornia.core import Tensor

from .architecture import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

__all__ = ['load_sam']


@dataclass
class SamPrediction:
    """To map the results of the `SamPredictor`

    Args:
        masks: Shape must be :math:`(B, K, H, W)` or :math:`(K, H, W)` where K is the number masks predicted.
        scores: Intersection Over Union for each prediction. Shape :math:`(B, K)` or :math:`(K)`.
        logits: These low resolution logits can be passed to a subsequent iteration as mask input. Shape of
                :math:`(B, K, H, W)` or :math:`(K, H, W)`, normally H=W=256.
    """

    masks: Tensor
    scores: Tensor
    logits: Tensor

    def drop(self, idx: int | slice | Tensor) -> SamPrediction:
        """Drop the passed index for all data.

        Performs `self.prop = self.prop[idx]` for each property
        """
        self.masks = self.masks[idx]
        self.scores = self.scores[idx]
        self.logits = self.logits[idx]

        return self


def load_sam_vit_h(checkpoint: str | None = None, device: torch.device | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(7, 15, 23, 31),
        checkpoint=checkpoint,
        device=device,
    )


def load_sam_vit_l(checkpoint: str | None = None, device: torch.device | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(5, 11, 17, 23),
        checkpoint=checkpoint,
        device=device,
    )


def load_sam_vit_b(checkpoint: str | None = None, device: torch.device | None = None) -> Sam:
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        checkpoint=checkpoint,
        device=device,
    )


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-6


def _build_sam(
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: tuple[int, ...],
    checkpoint: str | None = None,
    device: torch.device | None = None,
) -> Sam:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
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
        device=device,
    )

    sam.eval()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=device)
        sam.load_state_dict(state_dict)

    sam = sam.to(device=device)
    return sam


class SamType(Enum):
    """Map the SAM model types."""

    vit_h = 0
    vit_l = 1
    vit_b = 2


map_load_sam = {SamType.vit_h: load_sam_vit_h, SamType.vit_l: load_sam_vit_l, SamType.vit_b: load_sam_vit_b}
_map_sam_type = {'vit_h': SamType.vit_h, 'vit_l': SamType.vit_l, 'vit_b': SamType.vit_b}


def load_sam(model_type: str | int | SamType, checkpoint: str | None = None, device: torch.device | None = None) -> Sam:
    """Load a SAM model based on the model type.

    Args:
        model_type: the available models are:

            - 0, 'vit_h' or :func:`kornia.contrib.sam.SamType.vit_h`
            - 1, 'vit_l' or :func:`kornia.contrib.sam.SamType.vit_l`
            - 2, 'vit_b' or :func:`kornia.contrib.sam.SamType.vit_b`

        checkpoint: The filepath for the respective checkpoint
        device: The desired device to load the weights and move the model

    Returns:
        The respective SAM model

    Example:
        >>> # Input should be a RGB batched image
        >>> inpt = torch.randint(0, 255, (1, 3, 384, 384)).float()
        >>> inpt_after_resize = kornia.geometry.resize(inpt, (256, 256))
        >>> sam_model = load_sam('vit_b')
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
    if isinstance(model_type, int):
        model_type = SamType(model_type)
    elif isinstance(model_type, str):
        model_type = _map_sam_type[model_type]

    return map_load_sam[model_type](checkpoint, device)
