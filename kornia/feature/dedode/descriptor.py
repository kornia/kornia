import torch.nn.functional as F

from kornia.core import Module, Tensor


class DeDoDeDescriptor(Module):
    def __init__(self, encoder: Module, decoder: Module, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        features, sizes = self.encoder(images)
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            if idx == 0:
                descriptions, context = self.decoder(feature_map, scale=scale, context=context)
            else:
                delta_descriptions, context = self.decoder(feature_map, scale=scale, context=context)
                descriptions = descriptions + delta_descriptions
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptions = F.interpolate(descriptions, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)
        return descriptions
