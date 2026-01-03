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

"""Main SigLip2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import SigLip2Config
from .text_encoder import SigLip2TextModel
from .vision_encoder import SigLip2VisionModel

__all__ = ["SigLip2Model", "SigLip2Result"]


@dataclass
class SigLip2Result:
    """Result from SigLip2 model forward pass.

    Attributes:
        image_embeds: Image embeddings of shape (batch_size, projection_dim) or None.
        text_embeds: Text embeddings of shape (batch_size, projection_dim) or None.
        logit_scale: Logit scale parameter (temperature)
        logits_per_image: Logits for image-to-text matching of shape (batch_size, batch_size) or None.
        logits_per_text: Logits for text-to-image matching of shape (batch_size, batch_size) or None.
        loss: Contrastive loss or None.
    """

    logit_scale: torch.Tensor
    image_embeds: Optional[torch.Tensor] = None
    text_embeds: Optional[torch.Tensor] = None
    logits_per_image: Optional[torch.Tensor] = None
    logits_per_text: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class SigLip2Model(nn.Module):
    """SigLip2 vision-language model.

    This model combines a vision encoder and text encoder with projection layers
    to produce aligned embeddings in a shared space.

    Args:
        config: Model configuration.

    Note:
        Image preprocessing: Images should be preprocessed to match SigLip2ImagePreprocessor.

    Example:
        >>> import torch
        >>> from kornia.models.siglip2 import SigLip2Model, SigLip2Config, SigLip2ImagePreprocessor
        >>>
        >>> # Create model
        >>> config = SigLip2Config()
        >>> model = SigLip2Model(config)
        >>>
        >>> # Create preprocessor and process images
        >>> preprocessor = SigLip2ImagePreprocessor(image_size=(224, 224))
        >>> images = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.float32)
        >>> pixel_values = preprocessor(images)
        >>>
        >>> # Process image features
        >>> image_features = model.get_image_features(pixel_values)
        >>>
        >>> # Process text features
        >>> input_ids = torch.randint(0, 32000, (2, 10))
        >>> text_features = model.get_text_features(input_ids)
        >>>
        >>> # Joint processing
        >>> output = model(pixel_values=pixel_values, input_ids=input_ids)
        >>> logits = output.logits_per_image  # Image-text similarity scores
    """

    def __init__(self, config: SigLip2Config) -> None:
        super().__init__()
        self.config = config

        # vision and text encoders
        self.vision_model = SigLip2VisionModel(config.vision_config)
        self.text_model = SigLip2TextModel(config.text_config)

        # projection layers (use Identity when projection_dim == hidden_size)
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size
        projection_dim = config.projection_dim

        if projection_dim != vision_hidden_size:
            self.vision_projection = nn.Linear(vision_hidden_size, projection_dim)
        else:
            self.vision_projection = nn.Identity()

        if projection_dim != text_hidden_size:
            self.text_projection = nn.Linear(text_hidden_size, projection_dim)
        else:
            self.text_projection = nn.Identity()

        # logit scale (temperature parameter)
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

        # logit bias
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get image features.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).
            attention_mask: Optional attention mask for vision encoder of shape (batch_size, seq_len).
            normalize: Whether to normalize the output features.

        Returns:
            Image features of shape (batch_size, projection_dim).
        """
        vision_outputs = self.vision_model(pixel_values, attention_mask=attention_mask)
        image_features = vision_outputs[0]  # Pooled output

        # Apply projection (will be identity if projection_dim == hidden_size)
        image_features = self.vision_projection(image_features)

        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Get text features.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            position_ids: Optional position IDs of shape (batch_size, seq_len).
            normalize: Whether to normalize the output features.

        Returns:
            Text features of shape (batch_size, projection_dim).
        """
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_features = text_outputs[0]  # Pooled output

        # apply projection (will be identity if projection_dim == hidden_size)
        text_features = self.text_projection(text_features)

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> SigLip2Result:
        """Forward pass.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Optional attention mask for text encoder of shape (batch_size, seq_len).
            position_ids: Optional position IDs for text encoder.
            return_loss: Whether to compute and return contrastive loss.

        Returns:
            SigLip2Result containing:
            - image_embeds: Image embeddings (if pixel_values provided)
            - text_embeds: Text embeddings (if input_ids provided)
            - logit_scale: Logit scale parameter
            - logits_per_image: Logits for image-to-text matching (if both provided)
            - logits_per_text: Logits for text-to-image matching (if both provided)
            - loss: Contrastive loss (if return_loss=True and both provided)
        """
        # get embeddings
        image_embeds = self.get_image_features(pixel_values, normalize=True) if pixel_values is not None else None
        text_embeds = (
            self.get_text_features(input_ids, attention_mask=attention_mask, position_ids=position_ids, normalize=True)
            if input_ids is not None
            else None
        )

        logit_scale = self.logit_scale.exp()
        logits_per_image = None
        logits_per_text = None
        loss = None

        # compute similarity logits if both embeddings available
        if image_embeds is not None and text_embeds is not None:
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale + self.logit_bias
            logits_per_image = logits_per_text.t()

            # compute loss if requested for training or evaluation
            if return_loss:
                batch_size = image_embeds.shape[0]
                labels = torch.arange(batch_size, device=image_embeds.device)
                loss_img = -F.logsigmoid(logits_per_image[labels, labels]).mean()
                loss_txt = -F.logsigmoid(logits_per_text[labels, labels]).mean()
                loss = (loss_img + loss_txt) / 2.0

        return SigLip2Result(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logit_scale=logit_scale,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            loss=loss,
        )
