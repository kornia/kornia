from typing import Dict

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE

urls: Dict[str, str] = {}
urls["liberty"] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-liberty.params"  # pylint: disable
urls["notredame"] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-notredame.params"  # pylint: disable
urls["yosemite"] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-yosemite.params"  # pylint: disable


class TFeat(nn.Module):
    r"""Module, which computes TFeat descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Learning local feature descriptors
    with triplets and shallow convolutional neural networks".
    See :cite:`TFeat2016` for more details

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        torch.Tensor: TFeat descriptor of the patches.

    Shape:
        - Input: :math:`(B, 1, 32, 32)`
        - Output: :math:`(B, 128)`

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> tfeat = TFeat()
        >>> descs = tfeat(input) # 16x128
    """

    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh(),
        )
        self.descr = nn.Sequential(nn.Linear(64 * 8 * 8, 128), nn.Tanh())
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["liberty"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        KORNIA_CHECK_SHAPE(input, ["B", "1", "32", "32"])
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x
