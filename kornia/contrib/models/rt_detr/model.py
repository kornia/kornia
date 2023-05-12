from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_decoder import RTDETRDecoder
from kornia.core import Tensor


class RTDETRModelType(Enum):
    r50 = 0
    r101 = 1
    l = 2  # noqa: E741
    x = 3


@dataclass
class RTDETRConfig:
    model_type: str | int | RTDETRModelType | None = None
    checkpoint: str | None = None

    backbone_indices: tuple[int, ...] = (1, 2, 3)


class RTDETR(ModelBase[RTDETRConfig]):
    def __init__(self, backbone, neck, decoder_head, backbone_indices: tuple[int, ...]):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.decoder_head = decoder_head
        self.backbone_indices = backbone_indices

    @staticmethod
    def from_config(config: RTDETRConfig) -> RTDETR:
        model_type = config.model_type

        if isinstance(model_type, int):
            model_type = RTDETRModelType(model_type)
        elif isinstance(model_type, str):
            model_type = getattr(RTDETRModelType, model_type)

        if model_type == RTDETRModelType.r50:
            backbone = ResNetD([3, 4, 6, 3])
            neck = HybridEncoder(backbone.out_channels, 256)

        elif model_type == RTDETRModelType.r101:
            backbone = ResNetD([3, 4, 23, 3])
            neck = HybridEncoder(backbone.out_channels, 256)

        elif model_type == RTDETRModelType.l:
            raise NotImplementedError

        elif model_type == RTDETRModelType.x:
            raise NotImplementedError

        decoder_head = RTDETRDecoder(80, 256, 300, [256, 256, 256], 4, 8, 6)

        model = RTDETR(backbone, neck, decoder_head, config.backbone_indices)

        if config.checkpoint:
            model.load_checkpoint(config.checkpoint)
        return model

    def forward(self, images: Tensor):
        fmaps = self.backbone(images)
        fmaps = [fmap for i, fmap in enumerate(fmaps) if i in self.backbone_indices]
        fmaps = self.neck(images)
        bboxes, logits = self.decoder_head(images)
