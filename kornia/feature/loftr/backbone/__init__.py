from __future__ import annotations

from typing import Any

from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config: dict[str, Any]) -> ResNetFPN_8_2 | ResNetFPN_16_4:
    if config["backbone_type"] == "ResNetFPN":
        if config["resolution"] == (8, 2):
            return ResNetFPN_8_2(config["resnetfpn"])
        elif config["resolution"] == (16, 4):
            return ResNetFPN_16_4(config["resnetfpn"])

    raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} with res {config['resolution']} not supported.")
