"""Based from the original code from Meta Platforms, Inc. and affiliates.

https://github.com/facebookresearch/segment-
anything/blob/3518c86b78b3bc9cf4fbe3d18e682fad1c79dc51/segment_anything/modeling/sam.py
"""
from __future__ import annotations

from typing import Any

import torch

from kornia.contrib.sam.architecture.image_encoder import ImageEncoderViT
from kornia.contrib.sam.architecture.mask_decoder import MaskDecoder
from kornia.contrib.sam.architecture.prompt_encoder import PromptEncoder
from kornia.contrib.sam.base import SegmentationResults
from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class Sam(Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self, image_encoder: ImageEncoderViT, prompt_encoder: PromptEncoder, mask_decoder: MaskDecoder
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

    @torch.no_grad()
    def forward(
        self, images: Tensor, batched_prompts: list[dict[str, Any]], multimask_output: bool
    ) -> list[SegmentationResults]:
        """Predicts masks end-to-end from provided images and prompts.

        This method expects that the images have already been pre-processed, at least been normalized, resized and
        padded to be compatible with the `self.image_encoder`.

        .. note: For each image :math:`(3, H, W)`, it is possible to input a batch (:math:`K`) of :math:`N` prompts,
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
        KORNIA_CHECK_SHAPE(images, ['B', '3', 'H', 'W'])
        KORNIA_CHECK(
            images.shape[0] == len(batched_prompts),
            'The number of images (`B`) should match with the length of prompts!',
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
