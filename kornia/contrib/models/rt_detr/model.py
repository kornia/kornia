"""Based on code from PaddleDetection."""

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
    resnet50 = 0
    resnet101 = 1
    hgnetv2_l = 2
    hgnetv2_x = 3


@dataclass
class RTDETRConfig:
    """Configuration to construct RT-DETR model.

    Args:
        model_type: model variant. Available models are

            - ResNet-50
            - ResNet-101
            - L (HGNetV2)
            - X (HGNetV2)

        num_classes: number of classes
        checkpoint: URL or local path of model weights
        neck_hidden_dim: hidden dim for neck
        neck_dim_feedforward: feed-forward network dim for neck
        head_hidden_dim: hidden dim for head. Default: 256
        head_num_queries: number of queries for DETR transformer decoder. Default: 300
    """

    model_type: RTDETRModelType | str | int
    num_classes: int
    checkpoint: str | None = None

    neck_hidden_dim: int | None = None
    neck_dim_feedforward: int | None = None
    head_hidden_dim: int = 256
    head_num_queries: int = 300


class RTDETR(ModelBase[RTDETRConfig]):
    """RT-DETR Object Detection model, as described in https://arxiv.org/abs/2304.08069."""

    def __init__(self, backbone: ResNetD | PPHGNetV2, neck: HybridEncoder, head: RTDETRHead):
        """Construct RT-DETR Object Detection model.

        Args:
            backbone: backbone network for feature extraction
            neck: neck network for feature fusion
            head: head network to decode features into detection results
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @staticmethod
    def from_config(config: RTDETRConfig) -> RTDETR:
        """Construct RT-DETR Object Detection model from a config object.

        Args:
            config: configuration object for RT-DETR. Only ResNet-50, ResNet-101, L, and X are supported
        """
        model_type = config.model_type
        if isinstance(model_type, int):
            model_type = RTDETRModelType(model_type)
        elif isinstance(model_type, str):
            model_type = getattr(RTDETRModelType, model_type)

        backbone: ResNetD | PPHGNetV2

        if model_type == RTDETRModelType.resnet50:
            backbone = ResNetD.from_config(50)
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024

        elif model_type == RTDETRModelType.resnet101:
            backbone = ResNetD.from_config(101)
            neck_hidden_dim = config.neck_hidden_dim or 384
            neck_dim_feedforward = config.neck_dim_feedforward or 2048

        elif model_type == RTDETRModelType.hgnetv2_l:
            backbone = PPHGNetV2.from_config("L")
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024

        elif model_type == RTDETRModelType.hgnetv2_x:
            backbone = PPHGNetV2.from_config("X")
            neck_hidden_dim = config.neck_hidden_dim or 384
            neck_dim_feedforward = config.neck_dim_feedforward or 2038

        neck = HybridEncoder(backbone.out_channels, neck_hidden_dim, neck_dim_feedforward)
        head = RTDETRHead(
            config.num_classes, config.head_hidden_dim, config.head_num_queries, [neck_hidden_dim] * 3, 4, 8, 6
        )

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
        # box format is cxcywh
        # convert to xywh
        # bboxes[..., :2] -= bboxes[..., 2:] * 0.5  # in-place operation is not torch.compile()-friendly
        cxcy = bboxes[..., :2]
        wh = bboxes[..., 2:]
        bboxes = concatenate([cxcy - wh * 0.5, wh], -1)

        bboxes = bboxes * tensor([W, H, W, H], device=bboxes.device, dtype=bboxes.dtype).view(1, 1, 4)
        scores = logits.softmax(-1)[:, :, :-1]  # why the last class is removed?

        scores, labels = scores.max(-1)
        return DetectionResults(labels, scores, bboxes)
