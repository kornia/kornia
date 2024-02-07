import torch.nn.functional as F
from torch import nn


class DeDoDeDescriptor(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images,
    ):
        features, sizes = self.encoder(images)
        descriptions = 0
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptions, context = self.decoder(feature_map, scale=scale, context=context)
            descriptions = descriptions + delta_descriptions
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptions = F.interpolate(descriptions, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return descriptions
