from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import torch

from kornia.core import Module, Tensor
from kornia.geometry.transform import resize


# --------------------------------------------------------------------------------------------------------------------
# model base
# TODO: maybe this should live in another place
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class SegmentationResults:
    logits: Tensor
    scores: Tensor
    mask_threshold: float = 0.0

    @property
    def binary_masks(self) -> Tensor:
        """Binary mask generated from logits considering the mask_threshold.

        Shape will be the same of logits :math:`(B, C, H, W)` where :math:`C` is the number masks predicted.

        .. note: If you run `original_res_logits`, this will generate the masks based on the original resolution logits.
           Otherwise, this will use the low resolution logits (self.logits).
        """
        if self._original_res_logits is not None:
            x = self._original_res_logits
        else:
            x = self.logits

        return x > self.mask_threshold

    def original_res_logits(
        self, input_size: tuple[int, int], original_size: tuple[int, int], image_size_encoder: tuple[int, int] | None
    ) -> Tensor:
        """Remove padding and upscale the logits to the original image size.

        Resize to image encoder input -> remove padding (bottom and right) -> Resize to original size

        .. note: This method set a internal `original_res_logits` which will be used if available for the binary masks.

        Args:
            input_size: The size of the image input to the model, in (H, W) format. Used to remove padding.
            original_size: The original size of the image before resizing for input to the model, in (H, W) format.
            image_size_encoder: The size of the input image for image encoder, in (H, W) format. Used to resize the
                                logits back to encoder resolution before remove the padding.

        Returns:
            Batched logits in :math:`KxCxHxW` format, where (H, W) is given by original_size.
        """
        x = self.logits

        if isinstance(image_size_encoder, tuple):
            x = resize(x, size=image_size_encoder, interpolation='bilinear', align_corners=False, antialias=False)
        x = x[..., : input_size[0], : input_size[1]]

        x = resize(x, size=original_size, interpolation='bilinear', align_corners=False, antialias=False)

        self._original_res_logits = x
        return self._original_res_logits

    def squeeze(self, dim: int = 0) -> SegmentationResults:
        self.logits = self.logits.squeeze(dim)
        self.scores = self.scores.squeeze(dim)
        if isinstance(self._original_res_logits, Tensor):
            self._original_res_logits = self._original_res_logits.squeeze(dim)

        return self


ModelType = TypeVar('ModelType', bound=Enum)


class ModelBase(ABC, Module, Generic[ModelType]):
    """Abstract model class with some utilities function."""

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
    def build(model_type: str | int | ModelType) -> ModelBase[ModelType]:
        """This function should build the desired model type."""
        raise NotImplementedError

    @staticmethod
    def from_pretrained(
        model_type: str | int | ModelType, checkpoint: str | None = None, device: torch.device | None = None
    ) -> ModelBase[ModelType]:
        """This function should build the desired model type, load the checkpoint and move to device."""
        raise NotImplementedError
