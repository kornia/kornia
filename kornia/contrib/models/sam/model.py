"""Based from the original code from Meta Platforms, Inc. and affiliates.

https://github.com/facebookresearch/segment-
anything/blob/3518c86b78b3bc9cf4fbe3d18e682fad1c79dc51/segment_anything/build_sam.py

https://github.com/facebookresearch/segment-
anything/blob/3518c86b78b3bc9cf4fbe3d18e682fad1c79dc51/segment_anything/modeling/sam.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from kornia.contrib.models import SegmentationResults
from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.sam.architecture.common import LayerNorm
from kornia.contrib.models.sam.architecture.image_encoder import ImageEncoderViT
from kornia.contrib.models.sam.architecture.mask_decoder import MaskDecoder
from kornia.contrib.models.sam.architecture.prompt_encoder import PromptEncoder
from kornia.contrib.models.sam.architecture.transformer import TwoWayTransformer
from kornia.contrib.models.tiny_vit import TinyViT
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class SamModelType(Enum):
    """Map the SAM model types."""

    vit_h = 0
    vit_l = 1
    vit_b = 2
    mobile_sam = 3


@dataclass
class SamConfig:
    """Encapsulate the Config to build a SAM model.

    Args:
        model_type: the available models are:

            - 0, 'vit_h' or :func:`kornia.contrib.sam.SamModelType.vit_h`
            - 1, 'vit_l' or :func:`kornia.contrib.sam.SamModelType.vit_l`
            - 2, 'vit_b' or :func:`kornia.contrib.sam.SamModelType.vit_b`
            - 3, 'mobile_sam', or :func:`kornia.contrib.sam.SamModelType.mobile_sam`

        checkpoint: URL or a path for a file with the weights of the model
        encoder_embed_dim: Patch embedding dimension.
        encoder_depth: Depth of ViT.
        encoder_num_heads: Number of attention heads in each ViT block.
        encoder_global_attn_indexes: Encoder indexes for blocks using global attention.
    """

    model_type: Optional[str | int | SamModelType] = None
    checkpoint: Optional[str] = None
    pretrained: bool = False

    encoder_embed_dim: Optional[int] = None
    encoder_depth: Optional[int] = None
    encoder_num_heads: Optional[int] = None
    encoder_global_attn_indexes: Optional[tuple[int, ...]] = None


class Sam(ModelBase[SamConfig]):
    mask_threshold: float = 0.0

    def __init__(
        self, image_encoder: ImageEncoderViT | TinyViT, prompt_encoder: PromptEncoder, mask_decoder: MaskDecoder
    ) -> None:
        """SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder: The backbone used to encode the image into image embeddings that allow for efficient mask
                           prediction.
            prompt_encoder: Encodes various types of input prompts.
            mask_decoder: Predicts masks from the image embeddings and encoded prompts.
        """

        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    @staticmethod
    def from_name(name: str) -> Sam:
        """Build/load the SAM model based on it's name.

        Args:
            name: The name of the SAM model. Valid names are:
                - 'vit_b'
                - 'vit_l'
                - 'vit_h'
                - 'mobile_sam'

        Returns:
            The respective SAM model
        """
        if name in ["vit_b", "vit_l", "vit_h", "mobile_sam"]:
            return Sam.from_config(SamConfig(name))
        else:
            raise ValueError(f"Invalid SAM model name: {name}")

    @staticmethod
    def from_config(config: SamConfig) -> Sam:
        """Build/load the SAM model based on it's config.

        Args:
            config: The SamConfig data structure. If the model_type is available, build from it, otherwise will use
                    the parameters set.
        Returns:
            The respective SAM model

        Example:
            >>> from kornia.contrib.models.sam import SamConfig
            >>> sam_model = Sam.from_config(SamConfig('vit_b'))
        """
        model_type = config.model_type

        if isinstance(model_type, int):
            model_type = SamModelType(model_type)
        elif isinstance(model_type, str):
            _map_sam_type = {
                "vit_h": SamModelType.vit_h,
                "vit_l": SamModelType.vit_l,
                "vit_b": SamModelType.vit_b,
                "mobile_sam": SamModelType.mobile_sam,
            }
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

        elif model_type == SamModelType.mobile_sam:
            # TODO: merge this with _build_sam()
            prompt_embed_dim = 256
            image_size = 1024
            vit_patch_size = 16
            image_embedding_size = image_size // vit_patch_size

            model = Sam(
                image_encoder=TinyViT.from_config("5m", img_size=image_size, mobile_sam=True),
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
                #     pixel_mean=[123.675, 116.28, 103.53],
                #     pixel_std=[58.395, 57.12, 57.375],
            )

        elif (
            isinstance(config.encoder_embed_dim, int)
            and isinstance(config.encoder_depth, int)
            and isinstance(config.encoder_num_heads, int)
            and isinstance(config.encoder_global_attn_indexes, int)
        ):
            model = _build_sam(
                encoder_embed_dim=config.encoder_embed_dim,
                encoder_depth=config.encoder_depth,
                encoder_num_heads=config.num_heads,
                encoder_global_attn_indexes=config.encoder_global_attn_indexes,
            )

        else:
            raise NotImplementedError("Unexpected config. The model_type should be provide or the encoder configs.")

        checkpoint = config.checkpoint
        if config.pretrained:
            if checkpoint is None:
                checkpoint = {
                    SamModelType.vit_b: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    SamModelType.vit_l: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    SamModelType.vit_h: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    SamModelType.mobile_sam: "https://github.com/ChaoningZhang/MobileSAM/raw/a509aac54fdd7af59f843135f2f7cee307283c88/weights/mobile_sam.pt",
                }[model_type]
            else:
                warnings.warn("checkpoint is not None. pretrained=True is ignored")

        if checkpoint:
            model.load_checkpoint(checkpoint)

        return model

    @torch.no_grad()
    def forward(
        self, images: Tensor, batched_prompts: list[dict[str, Any]], multimask_output: bool
    ) -> list[SegmentationResults]:
        """Predicts masks end-to-end from provided images and prompts.

        This method expects that the images have already been pre-processed, at least been normalized, resized and
        padded to be compatible with the `self.image_encoder`.

        .. note:: For each image :math:`(3, H, W)`, it is possible to input a batch (:math:`K`) of :math:`N` prompts,
                 the results are batched by the number of prompts batch. So given a prompt with :math:`K=5`, and
                 :math:`N=10`, the results will look like :math:`5xCxHxW` where :math:`C` is determined by
                 multimask_output. And within each of these masks :math:`(5xC)`, it should be possible to find
                 :math:`N` instances if the model succeed.

        Args:
            images: The image as a torch tensor in :math:`(B, 3, H, W)` format, already transformed for input to the
                    model.
            batched_prompts: A list over the batch of images (list length should be :math:`B`), each a dictionary with
                             the following keys. If it does not have the respective prompt, it should not be included
                             in this dictionary. The options are:

                - "points": tuple of (Tensor, Tensor) within the coordinate keypoints and their respective labels.
                            the tuple should look like (keypoints, labels), where:

                            - The keypoints (a tensor) are a batched point prompts for this image, with shape
                              :math:`(K, N, 2)`. Already transformed to the input frame of the model.
                            - The labels (a tensor) are a batched labels for point prompts, with shape :math:`(K, N)`.
                              Where 1 indicates a foreground point and 0 indicates a background point.

                - "boxes": (Tensor) Batched box inputs, with shape :math:`(K, 4)`. Already transformed to the input
                           frame of the model.
                - "mask_inputs": (Tensor) Batched mask inputs to the model, in the form :math:`(K, 1, H, W)`.

            multimask_output: Whether the model should predict multiple disambiguating masks, or return a single mask.

        Returns:
            A list over input images, where each element is as SegmentationResults the following.
                - logits: Low resolution logits with shape :math:`(K, C, H, W)`. Can be passed as mask input to
                          subsequent iterations of prediction. Where :math:`K` is the number of input prompts,
                          :math:`C` is determined by multimask_output, and :math:`H=W=256` are the model output size.
                - scores: The model's predictions of mask quality (iou prediction), in shape BxC.
        """

        KORNIA_CHECK_SHAPE(images, ["B", "3", "H", "W"])
        KORNIA_CHECK(
            images.shape[0] == len(batched_prompts),
            "The number of images (`B`) should match with the length of prompts!",
        )

        image_embeddings = self.image_encoder(images)

        outputs = []
        for prompt_record, curr_embedding in zip(batched_prompts, image_embeddings):
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=prompt_record.get("points", None),
                boxes=prompt_record.get("boxes", None),
                masks=prompt_record.get("mask_inputs", None),
            )

            # Predict masks
            low_res_logits, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding[None, ...],
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # Save results
            outputs.append(SegmentationResults(low_res_logits, iou_predictions, self.mask_threshold))

        return outputs


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
        #     pixel_mean=[123.675, 116.28, 103.53],
        #     pixel_std=[58.395, 57.12, 57.375],
    )
