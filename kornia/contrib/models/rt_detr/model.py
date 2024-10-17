"""Based on code from PaddleDetection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from kornia.contrib.models.base import ModelBase
from kornia.contrib.models.rt_detr.architecture.hgnetv2 import PPHGNetV2
from kornia.contrib.models.rt_detr.architecture.hybrid_encoder import HybridEncoder
from kornia.contrib.models.rt_detr.architecture.resnet_d import ResNetD
from kornia.contrib.models.rt_detr.architecture.rtdetr_head import RTDETRHead
from kornia.core import Tensor

URLs = {
    "rtdetr_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
    "rtdetr_r34vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth",
    "rtdetr_r50vd_m": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth",
    "rtdetr_r50vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth",
    "rtdetr_r101vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth",
}


class RTDETRModelType(Enum):
    """Enum class that maps RT-DETR model type."""

    resnet18d = 0
    resnet34d = 1
    resnet50d = 2
    resnet101d = 3
    hgnetv2_l = 4
    hgnetv2_x = 5
    resnet50d_m = 6


@dataclass
class RTDETRConfig:
    """Configuration to construct RT-DETR model.

    Args:
        model_type: model variant. Available models are

            - ResNetD-18: ``0``, ``'resnet18d'`` or :attr:`RTDETRModelType.resnet18d`
            - ResNetD-34: ``1``, ``'resnet34d'`` or :attr:`RTDETRModelType.resnet34d`
            - ResNetD-50: ``2``, ``'resnet50d'`` or :attr:`RTDETRModelType.resnet50d`
            - ResNetD-101: ``3``, ``'resnet101d'`` or :attr:`RTDETRModelType.resnet101d`
            - HGNetV2-L: ``4``, ``'hgnetv2_l'`` or :attr:`RTDETRModelType.hgnetv2_l`
            - HGNetV2-X: ``5``, ``'hgnetv2_x'`` or :attr:`RTDETRModelType.hgnetv2_x`

        num_classes: number of classes.
        checkpoint: URL or local path of model weights.
        neck_hidden_dim: hidden dim for neck.
        neck_dim_feedforward: feed-forward network dim for neck.
        neck_expansion: expansion ratio for neck.
        head_hidden_dim: hidden dim for head.
        head_num_queries: number of queries for Deformable DETR transformer decoder.
        head_num_decoder_layers: number of decoder layers for Deformable DETR transformer decoder.
    """

    model_type: RTDETRModelType | str | int
    num_classes: int
    input_size: int = 640
    checkpoint: Optional[str] = None

    neck_hidden_dim: Optional[int] = None
    neck_dim_feedforward: Optional[int] = None
    neck_expansion: Optional[float] = None
    head_hidden_dim: int = 256
    head_num_queries: int = 300
    head_num_decoder_layers: Optional[int] = None
    confidence_threshold: float = 0.3

    @staticmethod
    def from_name(model_name: str, num_classes: int = 80) -> RTDETRConfig:
        """Load model without pretrained weights.

        Args:
            model_name: 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
        """

        if model_name == "rtdetr_r18vd":
            config = RTDETRConfig(RTDETRModelType.resnet18d, num_classes, input_size=640)
        elif model_name == "rtdetr_r34vd":
            config = RTDETRConfig(RTDETRModelType.resnet34d, num_classes, input_size=640)
        elif model_name == "rtdetr_r50vd_m":
            config = RTDETRConfig(RTDETRModelType.resnet50d_m, num_classes, input_size=640)
        elif model_name == "rtdetr_r50vd":
            config = RTDETRConfig(RTDETRModelType.resnet50d, num_classes, input_size=640)
        elif model_name == "rtdetr_r101vd":
            config = RTDETRConfig(RTDETRModelType.resnet101d, num_classes, input_size=640)
        else:
            raise ValueError

        return config


class RTDETR(ModelBase[RTDETRConfig]):
    """RT-DETR Object Detection model, as described in https://arxiv.org/abs/2304.08069."""

    def __init__(self, backbone: ResNetD | PPHGNetV2, encoder: HybridEncoder, decoder: RTDETRHead):
        """Construct RT-DETR Object Detection model.

        Args:
            backbone: backbone network for feature extraction.
            neck: neck network for feature fusion.
            head: head network to decode features into detection results.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def from_config(config: RTDETRConfig) -> RTDETR:
        """Construct RT-DETR Object Detection model from a config object.

        Args:
            config: configuration object for RT-DETR.

        .. note::
            For ``config.neck_hidden_dim``, ``config.neck_dim_feedforward``, ``config.neck_expansion``, and
            ``config.head_num_decoder_layers``, if they are ``None``, their values will be replaced with the
            default values depending on the ``config.model_type``. See the source code for the default values.
        """
        model_type = config.model_type
        if isinstance(model_type, int):
            model_type = RTDETRModelType(model_type)
        elif isinstance(model_type, str):
            model_type = getattr(RTDETRModelType, model_type)

        backbone: ResNetD | PPHGNetV2

        if model_type == RTDETRModelType.resnet18d:
            backbone = ResNetD.from_config(18)
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024
            head_num_decoder_layers = config.head_num_decoder_layers or 3
            neck_expansion = config.neck_expansion or 0.5

        elif model_type == RTDETRModelType.resnet34d:
            backbone = ResNetD.from_config(34)
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024
            head_num_decoder_layers = config.head_num_decoder_layers or 4
            neck_expansion = config.neck_expansion or 0.5

        elif model_type == RTDETRModelType.resnet50d:
            backbone = ResNetD.from_config(50)
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024
            head_num_decoder_layers = config.head_num_decoder_layers or 6
            neck_expansion = config.neck_expansion or 1.0

        elif model_type == RTDETRModelType.resnet50d_m:
            backbone = ResNetD.from_config(50)
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024
            head_num_decoder_layers = config.head_num_decoder_layers or 6
            neck_expansion = config.neck_expansion or 0.5

        elif model_type == RTDETRModelType.resnet101d:
            backbone = ResNetD.from_config(101)
            neck_hidden_dim = config.neck_hidden_dim or 384
            neck_dim_feedforward = config.neck_dim_feedforward or 2048
            head_num_decoder_layers = config.head_num_decoder_layers or 6
            neck_expansion = config.neck_expansion or 1.0

        elif model_type == RTDETRModelType.hgnetv2_l:
            backbone = PPHGNetV2.from_config("L")
            neck_hidden_dim = config.neck_hidden_dim or 256
            neck_dim_feedforward = config.neck_dim_feedforward or 1024
            head_num_decoder_layers = config.head_num_decoder_layers or 6
            neck_expansion = config.neck_expansion or 1.0

        elif model_type == RTDETRModelType.hgnetv2_x:
            backbone = PPHGNetV2.from_config("X")
            neck_hidden_dim = config.neck_hidden_dim or 384
            neck_dim_feedforward = config.neck_dim_feedforward or 2048
            head_num_decoder_layers = config.head_num_decoder_layers or 6
            neck_expansion = config.neck_expansion or 1.0

        model = RTDETR(
            backbone,
            HybridEncoder(backbone.out_channels, neck_hidden_dim, neck_dim_feedforward, neck_expansion),
            RTDETRHead(
                num_classes=config.num_classes,
                hidden_dim=config.head_hidden_dim,
                num_queries=config.head_num_queries,
                in_channels=[neck_hidden_dim] * 3,
                num_decoder_layers=head_num_decoder_layers,
            ),
        )

        if config.checkpoint:
            model.load_checkpoint(config.checkpoint)
        return model

    @staticmethod
    def from_pretrained(model_name: str) -> RTDETR:
        """Load model from pretrained weights.

        Args:
            model_name: 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
        """

        if model_name not in URLs:
            raise ValueError(f"No pretrained model for '{model_name}'. Please select from {list(URLs.keys())}.")

        state_dict = torch.hub.load_state_dict_from_url(
            URLs[model_name], map_location="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        def map_name(old_name: str) -> str:
            new_name = old_name

            # Encoder renaming
            new_name = re.sub("encoder.pan_blocks", "encoder.ccfm.pan_blocks", new_name)
            new_name = re.sub("encoder.downsample_convs", "encoder.ccfm.downsample_convs", new_name)
            new_name = re.sub("encoder.fpn_blocks", "encoder.ccfm.fpn_blocks", new_name)
            new_name = re.sub("encoder.lateral_convs", "encoder.ccfm.lateral_convs", new_name)

            # Backbone renaming
            new_name = re.sub(".branch2b.", ".convs.branch2b.", new_name)
            new_name = re.sub(".branch2a.", ".convs.branch2a.", new_name)
            new_name = re.sub(".branch2c.", ".convs.branch2c.", new_name)

            return new_name

        def _state_dict_proc(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
            state_dict = state_dict["ema"]["module"]  # type:ignore
            new_state_dict = {}

            # Apply the regex-based mapping function to each key
            for old_name in state_dict.keys():
                new_name = map_name(old_name)
                new_state_dict[new_name] = state_dict[old_name]

            return new_state_dict

        model = RTDETR.from_name(model_name, num_classes=80)

        model.load_state_dict(_state_dict_proc(state_dict))
        return model

    @staticmethod
    def from_name(model_name: str, num_classes: int = 80) -> RTDETR:
        """Load model without pretrained weights.

        Args:
            model_name: 'rtdetr_r18vd', 'rtdetr_r34vd', 'rtdetr_r50vd_m', 'rtdetr_r50vd', 'rtdetr_r101vd'.
        """
        model = RTDETR.from_config(RTDETRConfig.from_name(model_name, num_classes))
        return model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Detect objects in an image.

        Args:
            images: images to be detected. Shape :math:`(N, C, H, W)`.

        Returns:
            - **logits** - Tensor of shape :math:`(N, Q, K)`, where :math:`Q` is the number of queries,
              :math:`K` is the number of classes.
            - **boxes** - Tensor of shape :math:`(N, Q, 4)`, where :math:`Q` is the number of queries.
        """

        feats = self.backbone(images)
        feats_buf = self.encoder(feats)
        logits, boxes = self.decoder(feats_buf)
        return logits, boxes
