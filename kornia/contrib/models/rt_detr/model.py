from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kornia.contrib.models import DetectionResults
from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.core import Tensor, concatenate, tensor


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
    def __init__(self, backbone, neck, head, backbone_indices: tuple[int, ...]):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
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

        decoder_head = RTDETRHead(80, 256, 300, [256, 256, 256], 4, 8, 6)

        model = RTDETR(backbone, neck, decoder_head, config.backbone_indices)

        if config.checkpoint:
            model.load_checkpoint(config.checkpoint)
        return model

    def forward(self, images: Tensor) -> DetectionResults:
        H, W = images.shape[2:]
        fmaps = self.backbone(images)
        fmaps = [fmap for i, fmap in enumerate(fmaps) if i in self.backbone_indices]
        fmaps = self.neck(fmaps)
        bboxes, logits = self.head(fmaps)

        # https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/post_process.py#L446
        cxcy, wh = bboxes.chunk(2, -1)
        bboxes = concatenate([cxcy - wh * 0.5, cxcy + wh * 0.5], -1)

        bboxes = bboxes * tensor([H, W, H, W], device=bboxes.device, dtype=bboxes.dtype).view(1, 1, 4)
        scores = logits.softmax(-1)[:, :, :-1]

        scores, labels = scores.max(-1)
        return DetectionResults(labels, scores, bboxes)
