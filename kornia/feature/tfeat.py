from typing import Dict

import torch
import torch.nn as nn

urls: Dict[str, str] = {}
urls[
    "liberty"
] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-liberty.params"  # noqa pylint: disable
urls[
    "notredame"
] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-notredame.params"  # noqa pylint: disable
urls[
    "yosemite"
] = "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-yosemite.params"  # noqa pylint: disable


class TFeat(nn.Module):
    r"""Module, which computes TFeat descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Learning local feature descriptors
    with triplets and shallow convolutional neural networks".
    See :cite:`TFeat2016` for more details

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        TFeat descriptor of the patches.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> tfeat = TFeat()
        >>> descs = tfeat(input) # 16x128
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(TFeat, self).__init__()
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
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['liberty'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict, strict=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.descr(x)
        return x
