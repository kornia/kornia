from __future__ import annotations

from typing import Any

from torch import no_grad
from torch.nn import functional as F

from kornia.core import Device, Module, Tensor, pad, stack, tensor

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


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
        device: Device = None,
    ) -> None:
        """SAM predicts object masks from an image and input prompts.

        Args:
            image_encoder: The backbone used to encode the image into image embeddings that allow for efficient mask
            prediction.
            prompt_encoder: Encodes various types of input prompts.
            mask_decoder: Predicts masks from the image embeddings and encoded prompts.
            pixel_mean: Mean values for normalizing pixels in the input image.
            pixel_std: Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", tensor(pixel_mean, device=device).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", tensor(pixel_std, device=device).view(-1, 1, 1), False)
        self._device = device

    @property
    def device(self) -> Device:
        return self._device

    @no_grad()
    def forward(self, batched_input: list[dict[str, Any]], multimask_output: bool) -> list[dict[str, Tensor]]:
        """Predicts masks end-to-end from provided images and prompts. If prompts are not known in advance, using
        SamPredictor is recommended over calling the model directly.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys. A prompt key can be
            excluded if it is not present. The options are:
                - 'image': The image as a torch tensor in 3xHxW format, already transformed for input to the model.
                - 'original_size': (tuple(int, int)) The original size of the image before transformation, as (H, W).
                - 'point_coords': (Tensor) Batched point prompts for this image, with shape BxNx2. Already
                transformed to the input frame of the model.
                - 'point_labels': (Tensor) Batched labels for point prompts, with shape BxN.
                - 'boxes': (Tensor) Batched box inputs, with shape Bx4. Already transformed to the input frame of the
                model.
                - 'mask_inputs': (Tensor) Batched mask inputs to the model, in the form Bx1xHxW.
            multimask_output: Whether the model should predict multiple disambiguating masks, or return a single mask.

        Returns:
            A list over input images, where each element is as dictionary with the following keys.
                - 'masks': (Tensor) Batched binary mask predictions, with shape BxCxHxW, where B is the number
                of input prompts, C is determiend by multimask_output, and (H, W) is the original size of the image.
                - 'iou_predictions': (Tensor) The model's predictions of mask quality, in shape BxC.
                - 'low_res_logits': (Tensor) Low resolution logits with shape BxCxHxW, where H=W=256. Can be passed as
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
