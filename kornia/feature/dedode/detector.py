import torch.nn.functional as F
from torch import nn


class DeDoDeDetector(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images,
    ):
        features, sizes = self.encoder(images)
        logits = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(feature_map, context=context, scale=scale)
            logits = logits + delta_logits.float()  # ensure float (need bf16 doesnt have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        return logits
