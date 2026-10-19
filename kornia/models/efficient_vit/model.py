# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch

from kornia.core.download import load_state_dict_from_url
from kornia.models.base import ModelBase
from kornia.models.efficient_vit import backbone as vit


def _get_base_url(model_type: Literal["b1", "b2", "b3"] = "b1", resolution: Literal[224, 256, 288] = 224) -> str:
    """Return the base URL of the model weights."""
    return f"https://huggingface.co/kornia/efficientvit_imagenet_{model_type}_r{resolution}/resolve/main/{model_type}-r{resolution}.pt"


@dataclass
class EfficientViTConfig:
    """Configuration to construct EfficientViT model.

    Model weights can be loaded from a checkpoint URL or local path.
    The model weights are hosted on HuggingFace's model hub: https://huggingface.co/kornia.

    Args:
        checkpoint: URL or local path of model weights. :meth:`EfficientViT.from_config` reads the architecture from
            the file name, which must start with the model type and a ``-`` (``b0``, ``b1``, ``b2``, ``b3``, ``l0``,
            ``l1``, ``l2`` or ``l3``), as in ``b1-r224.pt``: ``my_efficientvit.pt`` raises ``ValueError``.

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
    """EfficientViT backbone model.

    The pretrained backbones were trained on images normalized with the ImageNet mean and standard deviation, so a
    :math:`[0, 1]` image goes in as
    ``kornia.enhance.normalize(image, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))``.
    """

    def __init__(self, backbone: vit.EfficientViTBackbone | vit.EfficientViTLargeBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def from_config(config: EfficientViTConfig) -> EfficientViT:
        """Build the EfficientViT model from a configuration object.

        Args:
            config: EfficientViT configuration object. See :class:`EfficientViTConfig`.

        Returns:
            EfficientViT: the EfficientViT model, in evaluation mode, since it always loads ``config.checkpoint``
            (see :ref:`pretrained-models-conventions`).

        """
        # load the model from the checkpoint; expand ``~`` first, or a ``~`` path reaches the URL loader, which
        # resolves a file name already in the hub cache to that cached file instead of the local one
        checkpoint = os.path.expanduser(config.checkpoint)
        try:
            if os.path.isfile(checkpoint):
                with open(checkpoint, "rb") as f:
                    model_file = torch.load(f, map_location="cpu", weights_only=True)
            else:
                model_file = load_state_dict_from_url(config.checkpoint, map_location="cpu")
            model_file = model_file.get("state_dict", model_file)
        except RuntimeError as exc:
            raise RuntimeError(f"Unable to load the model from {config.checkpoint}.") from exc

        file_name = os.path.basename(checkpoint)
        model_type = file_name.split("-")[0]

        if model_type not in ["b0", "b1", "b2", "b3", "l0", "l1", "l2", "l3"]:
            raise ValueError(f"Unknown model type: {model_type}.")

        model = getattr(vit, f"efficientvit_backbone_{model_type}")()
        # The hosted checkpoints hold the backbone under a "backbone." prefix, next to a "head." classifier the
        # backbone has no slot for. A state dict saved from the backbone itself has no prefix and loads as it is.
        state_dict = {key[len("backbone.") :]: val for key, val in model_file.items() if key.startswith("backbone.")}
        model.load_state_dict(state_dict or model_file, strict=True)

        return EfficientViT(backbone=model).eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from the input images.

        Args:
            images: input images tensor of shape :math:`(B, C, H, W)`.

        Returns:
            a dictionary with the input under ``"input"``, the output of each stage under ``"stage0"``,
            ``"stage1"``, and so on, and the last stage's output again under ``"stage_final"``.

        """
        return self.backbone(images)
