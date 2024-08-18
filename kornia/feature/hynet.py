from typing import Dict

import torch
from torch import nn

from kornia.core import Module, Parameter, Tensor, tensor, zeros

urls: Dict[str, str] = {}
urls["liberty"] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_LIB.pth"  # pylint: disable
urls["notredame"] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_ND.pth"  # pylint: disable
urls["yosemite"] = "https://github.com/ducha-aiki/Key.Net-Pytorch/raw/main/model/HyNet/weights/HyNet_YOS.pth"  # pylint: disable


class FilterResponseNorm2d(Module):
    r"""Feature Response Normalization layer from 'Filter Response Normalization Layer: Eliminating Batch Dependence
    in the Training of Deep Neural Networks', see :cite:`FRN2019` for more details.

    .. math::
        y =  \gamma \times \frac{x}{\sqrt{\mathrm{E}[x^2]} + |\epsilon|} + \beta


    Args:
        num_features: number of channels
        eps: normalization constant
        is_bias: use bias
        is_scale: use scale
        drop_rate: dropout rate,
        is_eps_leanable: if eps is learnable

    Returns:
        torch.Tensor: Normalized features

    Shape:
        - Input: :math:`(B, \text{num_features}, H, W)`
        - Output: :math:`(B, \text{num_features}, H, W)`
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        is_bias: bool = True,
        is_scale: bool = True,
        is_eps_leanable: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = Parameter(torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.bias = Parameter(zeros(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = Parameter(tensor(1), requires_grad=True)
        else:
            self.register_buffer("eps", tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self) -> str:
        return "num_features={num_features}, eps={init_eps}".format(**self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x


class TLU(Module):
    r"""TLU layer from 'Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep
    Neural Networks, see :cite:`FRN2019` for more details. :math:`{\tau}` is learnable per channel.

    .. math::
        y = \max(x, {\tau})

    Args:
        num_features: number of channels

    Returns:
        torch.Tensor

    Shape:
        - Input: :math:`(B, \text{num_features}, H, W)`
        - Output: :math:`(B, \text{num_features}, H, W)`
    """

    def __init__(self, num_features: int) -> None:
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super().__init__()
        self.num_features = num_features
        self.tau = Parameter(-torch.ones(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self) -> str:
        return "num_features={num_features}".format(**self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, self.tau)


class HyNet(Module):
    r"""Module, which computes HyNet descriptors of given grayscale patches of 32x32.

    This is based on the original code from paper
    "HyNet: Learning Local Descriptor with Hybrid Similarity Measure and Triplet Loss".
    See :cite:`hynet2020` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.
        is_bias: use bias in TLU layers
        is_bias_FRN:  use bias in FRN layers
        dim_desc: descriptor dimensionality,
        drop_rate: dropout rate,
        eps_l2_norm: to avoid div by zero

    Returns:
        HyNet descriptor of the patches.

    Shape:
        - Input: :math:`(B, 1, 32, 32)`
        - Output: :math:`(B, 128)`

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> hynet = HyNet()
        >>> descs = hynet(input) # 16x128
    """

    patch_size = 32

    def __init__(
        self,
        pretrained: bool = False,
        is_bias: bool = True,
        is_bias_FRN: bool = True,
        dim_desc: int = 128,
        drop_rate: float = 0.3,
        eps_l2_norm: float = 1e-10,
    ) -> None:
        super().__init__()
        self.eps_l2_norm = eps_l2_norm
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FilterResponseNorm2d(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FilterResponseNorm2d(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FilterResponseNorm2d(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FilterResponseNorm2d(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False),
        )

        self.desc_norm = nn.LocalResponseNorm(2 * self.dim_desc, 2.0 * self.dim_desc, 0.5, 0.0)
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["liberty"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.desc_norm(x + self.eps_l2_norm)
        x = x.view(x.size(0), -1)
        return x
