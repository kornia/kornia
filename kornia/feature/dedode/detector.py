import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor


class DeDoDeDetector(nn.Module):
    def __init__(self, encoder: Module, decoder: Module, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        dtype = images.dtype
        features, sizes = self.encoder(images)
        context = None
        logits = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_logits, context = self.decoder(feature_map, context=context, scale=scale)
            if logits is None:
                logits = delta_logits
            else:
                logits = logits + delta_logits.float()  # ensure float (need bf16 doesn't have f.interpolate)
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                logits = F.interpolate(logits, size=size, mode="bicubic", align_corners=False)
                context = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        return logits.to(dtype)  # type: ignore
