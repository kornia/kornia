from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial

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
        self.preds_iou = self.low_res_masks[idx]
        self.low_res_masks = self.low_res_masks[idx]

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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
    vit_h = 0
    vit_l = 1
    vit_n = 2


map_load_sam = {SamType.vit_h: load_sam_vit_h, SamType.vit_l: load_sam_vit_l, SamType.vit_n: load_sam_vit_b}
_map_sam_type = {'vit_h': SamType.vit_h, 'vit_l': SamType.vit_l, 'vit_n': SamType.vit_n}


def load_sam(model_type: str | int | SamType, checkpoint: str | None = None, device: torch.device | None = None) -> Sam:
    """Load a SAM model based on the model type.

    Args:
        model_type: the available models are:
                - 0, 'vit_h' or `SamType.vit_h`
                - 1, 'vit_l' or `SamType.vit_l`
                - 2, 'vit_b' or `SamType.vit_b`
        checkpoint: The filepath for the respective checkpoint

    Returns:
        The respective SAM model
    """
    if isinstance(model_type, int):
        model_type = SamType(model_type)
    elif isinstance(model_type, str):
        model_type = _map_sam_type[model_type]

    return map_load_sam[model_type](checkpoint, device)
