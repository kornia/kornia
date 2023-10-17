from __future__ import annotations

from dataclasses import dataclass

import torch

from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.efficient_vit import backbone as vit
from kornia.core import Tensor


@dataclass
class EfficientViTConfig:
    """Configuration to construct EfficientViT model.

    Args:
        checkpoint: URL or local path of model weights.
    """

    checkpoint: str = "https://huggingface.co/kornia/efficientvit_imagenet_b1_r224.pt/resolve/main/b1-r224.pt"


class EfficientViT(ModelBase[EfficientViTConfig]):
    """EfficientViT backbone model."""

    def __init__(self, backbone: vit.EfficientViTBackbone | vit.EfficientViTLargeBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def from_config(config: EfficientViTConfig) -> EfficientViT:
        """Build the EfficientViT model from a configuration object.

        Args:
            config: EfficientViT configuration object.

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
