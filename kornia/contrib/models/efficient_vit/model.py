from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.efficient_vit import backbone as vit
from kornia.core import Tensor


def _get_base_url(model_type: Literal["b1", "b2", "b3"] = "b1", resolution: Literal[224, 256, 288] = 224) -> str:
    """Return the base URL of the model weights."""
    return f"https://huggingface.co/kornia/efficientvit_imagenet_{model_type}_r{resolution}/resolve/main/{model_type}-r{resolution}.pt"


@dataclass
class EfficientViTConfig:
    """Configuration to construct EfficientViT model.

    Model weights can be loaded from a checkpoint URL or local path.
    The model weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

    Args:
        checkpoint: URL or local path of model weights.
    """

    checkpoint: str = field(default_factory=_get_base_url)

    @classmethod
    def from_pretrained(
        cls, model_type: Literal["b1", "b2", "b3"], resolution: Literal[224, 256, 288]
    ) -> EfficientViTConfig:
        """Return a configuration object from a pre-trained model.

        Args:
            model_type: model type, one of :obj:`"b1"`, :obj:`"b2"`, :obj:`"b3"`.
            resolution: input resolution, one of :obj:`224`, :obj:`256`, :obj:`288`.
        """
        return cls(checkpoint=_get_base_url(model_type=model_type, resolution=resolution))


class EfficientViT(ModelBase[EfficientViTConfig]):
    """EfficientViT backbone model."""

    def __init__(self, backbone: vit.EfficientViTBackbone | vit.EfficientViTLargeBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def from_config(config: EfficientViTConfig) -> EfficientViT:
        """Build the EfficientViT model from a configuration object.

        Args:
            config: EfficientViT configuration object. See :class:`EfficientViTConfig`.

        Returns:
            EfficientViT: the EfficientViT model.
        """
        # load the model from the checkpoint
        try:
            model_file = torch.hub.load_state_dict_from_url(config.checkpoint, map_location="cpu")
            model_file = model_file["state_dict"] if "state_dict" in model_file else model_file
        except RuntimeError:
            raise RuntimeError(f"Unable to load the model from {config.checkpoint}.")

        file_name = config.checkpoint.split("/")[-1]
        model_type = file_name.split("-")[0]

        if model_type not in ["b0", "b1", "b2", "b3", "l0", "l1", "l2", "l3"]:
            raise ValueError(f"Unknown model type: {model_type}.")

        # create and load the model weights without strict until we polish the model files
        model = getattr(vit, f"efficientvit_backbone_{model_type}")()
        model.load_state_dict(model_file, strict=False)

        return EfficientViT(backbone=model)

    def forward(self, images: Tensor) -> Tensor:
        """Extract features from the input images.

        Args:
            images: input images tensor of shape :math:`(B, C, H, W)`.

        Returns:
            Dict[str, Tensor]: a dictionary containing the features.
        """
        feats = self.backbone(images)
        return feats
