from typing import Dict

import torch

import kornia


class HardNet(kornia.nn.HardNet):
    """
    Module, which computes HardNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Working hard to know your neighbor's
    margins: Local descriptor learning loss". See :cite:`HardNet2017` for more details.

    Args:
        pretrained: (bool) Download and set pretrained weights to the model. Default: false.

    Returns:
        torch.Tensor: HardeNet descriptor of the patches.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = kornia.feature.HardNet()
        >>> descs = hardnet(input) # 16x128
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(HardNet, self).__init__()
        kornia.deprecation_warning(
            "kornia.feature.HardNet", "kornia.nn.HardNet")
