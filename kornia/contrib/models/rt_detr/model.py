from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kornia.contrib.models import DetectionResults
from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
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


class RTDETR(ModelBase[RTDETRConfig]):
    def __init__(self, backbone: ResNetD | PPHGNetV2, neck: HybridEncoder, head: RTDETRHead):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @staticmethod
    def from_config(config: RTDETRConfig) -> RTDETR:
        model_type = config.model_type

        if isinstance(model_type, int):
            model_type = RTDETRModelType(model_type)
        elif isinstance(model_type, str):
            model_type = getattr(RTDETRModelType, model_type)

        if model_type == RTDETRModelType.r50:
            backbone = ResNetD.from_config(50)
            hidden_dim = 256
            ff_dim = 1024

        elif model_type == RTDETRModelType.r101:
            backbone = ResNetD.from_config(101)
            hidden_dim = 384
            ff_dim = 2048

        elif model_type == RTDETRModelType.l:
            backbone = PPHGNetV2.from_config("L")
            hidden_dim = 256
            ff_dim = 1024

        elif model_type == RTDETRModelType.x:
            backbone = PPHGNetV2.from_config("X")
            hidden_dim = 384
            ff_dim = 2048

        else:
            raise ValueError

        neck = HybridEncoder(backbone.out_channels, hidden_dim, ff_dim)
        head = RTDETRHead(80, 256, 300, [hidden_dim] * 3, 4, 8, 6)

        model = RTDETR(backbone, neck, head)

        if config.checkpoint:
            model.load_checkpoint(config.checkpoint)
        return model

    def forward(self, images: Tensor) -> DetectionResults:
        H, W = images.shape[2:]
        fmaps = self.backbone(images)
        fmaps = self.neck(fmaps)
        bboxes, logits = self.head(fmaps)

        # https://github.com/PaddlePaddle/PaddleDetection/blob/5d1f888362241790000950e2b63115dc8d1c6019/ppdet/modeling/post_process.py#L446
        cxcy, wh = bboxes.chunk(2, -1)
        bboxes = concatenate([cxcy - wh * 0.5, cxcy + wh * 0.5], -1)

        bboxes = bboxes * tensor([W, H, W, H], device=bboxes.device, dtype=bboxes.dtype).view(1, 1, 4)
        scores = logits.softmax(-1)[:, :, :-1]

        scores, labels = scores.max(-1)
        return DetectionResults(labels, scores, bboxes)
