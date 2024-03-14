"""Based from the original code from Meta Platforms, Inc. and affiliates.

https://github.com/facebookresearch/segment-
anything/blob/3518c86b78b3bc9cf4fbe3d18e682fad1c79dc51/segment_anything/modeling/mask_decoder.py
"""

from __future__ import annotations

import torch
from torch import nn

from kornia.contrib.models.common import MLP, LayerNorm2d
from kornia.core import Module, Tensor, concatenate, stack


class MaskDecoder(Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: Module,
        num_multimask_outputs: int = 3,
        activation: type[Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """Predicts masks given an image and prompt embeddings, using a transformer architecture.

        Args:
            transformer_dim: the channel dimension of the transformer
            transformer: the transformer used to predict masks
            num_multimask_outputs: the number of masks to predict when disambiguating masks
            activation: the type of activation to use when upscaling masks
            iou_head_depth: the depth of the MLP used to predict mask quality
            iou_head_hidden_dim: the hidden dimension of the MLP used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for i in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        """Predict masks given image and prompt embeddings.

        Args:
            image_embeddings: the embeddings from the image encoder
            image_pe: positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings: the embeddings of the points and boxes
            dense_prompt_embeddings: the embeddings of the mask inputs
            multimask_output: Whether to return multiple masks or a single mask.

        Returns:
            batched predicted masks
            batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predicts masks.

        See 'forward' for more details.
        """
        # Concatenate output tokens
        output_tokens = concatenate([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens[None, ...].expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = concatenate((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
