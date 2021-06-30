from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

urls: Dict[str, str] = {}
urls["hardnet++"] = "https://github.com/DagnyT/hardnet/raw/master/pretrained/pretrained_all_datasets/HardNet++.pth"
urls[
    "liberty_aug"
] = "https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth"  # noqa pylint: disable
urls["hardnet8v2"] = "http://cmp.felk.cvut.cz/~mishkdmy/hardnet8v2.pt"  # noqa pylint: disable


class HardNet(nn.Module):
    r"""Module, which computes HardNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Working hard to know your neighbor's
    margins: Local descriptor learning loss". See :cite:`HardNet2017` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        HardNet descriptor of the patches.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = HardNet()
        >>> descs = hardnet(input) # 16x128
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['liberty_aug'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict['state_dict'], strict=True)

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        "Utility function that normalizes the input by batch." ""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


class HardNet8(nn.Module):
    r"""Module, which computes HardNet8 descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper "Improving the HardNet Descriptor".
    See :cite:`HardNet2020` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        HardNet8 descriptor of the patches.

    Shape:
        - Input: (B, 1, 32, 32)
        - Output: (B, 128)

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hardnet = HardNet8()
        >>> descs = hardnet(input) # 16x128
    """

    def __init__(self, pretrained: bool = False):
        super(HardNet8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(self.weights_init)
        self.register_buffer('components', torch.ones(512, 128, dtype=torch.float))
        self.register_buffer('mean', torch.zeros(512, dtype=torch.float))

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['hardnet8v2'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict, strict=True)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        "Utility function that normalizes the input by batch." ""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(input)
        x_features: torch.Tensor = self.features(x_norm)
        mean: torch.Tensor = torch.jit.annotate(torch.Tensor, self.mean)
        components: torch.Tensor = torch.jit.annotate(torch.Tensor, self.components)
        x_prePCA = F.normalize(x_features.view(x_features.size(0), -1))
        pca = torch.mm(x_prePCA - mean, components)
        return F.normalize(pca, dim=1)
