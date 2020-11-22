from typing import Dict

import torch
import torch.nn as nn

import kornia


class SOSNet(kornia.nn.SOSNet):
    """128-dimensional SOSNet model definition for 32x32 patches.

    This is based on the original code from paper
    "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning".

    Args:
        pretrained (bool): Download and set pretrained weights to the model. Default: false.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(8, 1, 32, 32)
        >>> sosnet = kornia.feature.SOSNet()
        >>> descs = sosnet(input) # 8x128
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(SOSNet, self).__init__(pretrained)
        kornia.deprecation_warning("kornia.feature.SOSNet", "kornia.nn.SOSNet")
