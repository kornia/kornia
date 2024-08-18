from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import TypedDict

from kornia.core import Module, Tensor, concatenate
from kornia.filters import SpatialGradient
from kornia.geometry.transform import pyrdown

from .scale_space_detector import Detector_config, MultiResolutionDetector, get_default_detector_config


class KeyNet_conf(TypedDict):
    num_filters: int
    num_levels: int
    kernel_size: int
    Detector_conf: Detector_config


keynet_default_config: KeyNet_conf = {
    # Key.Net Model
    "num_filters": 8,
    "num_levels": 3,
    "kernel_size": 5,
    # Extraction Parameters
    "Detector_conf": get_default_detector_config(),
}

KeyNet_URL = "https://github.com/axelBarroso/Key.Net-Pytorch/raw/main/model/weights/keynet_pytorch.pth"


class _FeatureExtractor(Module):
    """Helper class for KeyNet.

    It loads both, the handcrafted and learnable blocks
    """

    def __init__(self) -> None:
        super().__init__()

        self.hc_block = _HandcraftedBlock()
        self.lb_block = _LearnableBlock()

    def forward(self, x: Tensor) -> Tensor:
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class _HandcraftedBlock(Module):
    """Helper class for KeyNet, it defines the handcrafted filters within the Key.Net handcrafted block."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial_gradient = SpatialGradient("sobel", 1)

    def forward(self, x: Tensor) -> Tensor:
        sobel = self.spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = self.spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = self.spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = concatenate([dx, dy, dx**2.0, dy**2.0, dx * dy, dxy, dxy**2.0, dxx, dyy, dxx * dyy], 1)

        return hc_feats


class _LearnableBlock(nn.Sequential):
    """Helper class for KeyNet.

    It defines the learnable blocks within the Key.Net
    """

    def __init__(self, in_channels: int = 10) -> None:
        super().__init__()

        self.conv0 = _KeyNetConvBlock(in_channels)
        self.conv1 = _KeyNetConvBlock()
        self.conv2 = _KeyNetConvBlock()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2(self.conv1(self.conv0(x)))
        return x


def _KeyNetConvBlock(
    in_channels: int = 8,
    out_channels: int = 8,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 2,
    dilation: int = 1,
) -> nn.Sequential:
    """Helper function for KeyNet.

    Default learnable convolutional block for KeyNet.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class KeyNet(Module):
    """Key.Net model definition -- local feature detector (response function). This is based on the original code
    from paper "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters". See :cite:`KeyNet2019` for
    more details.

    .. image:: _static/img/KeyNet.png

    Args:
        pretrained: Download and set pretrained weights to the model.
        keynet_conf: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.

    Returns:
        KeyNet response score.

    Shape:
        - Input: :math:`(B, 1, H, W)`
        - Output: :math:`(B, 1, H, W)`
    """

    def __init__(self, pretrained: bool = False, keynet_conf: KeyNet_conf = keynet_default_config) -> None:
        super().__init__()

        num_filters = keynet_conf["num_filters"]
        self.num_levels = keynet_conf["num_levels"]
        kernel_size = keynet_conf["kernel_size"]
        padding = kernel_size // 2

        self.feature_extractor = _FeatureExtractor()
        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters * self.num_levels, out_channels=1, kernel_size=kernel_size, padding=padding
            ),
            nn.ReLU(inplace=True),
        )
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(KeyNet_URL, map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict["state_dict"], strict=True)
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """
        x - input image
        """
        shape_im = x.shape
        feats: List[Tensor] = [self.feature_extractor(x)]
        for i in range(1, self.num_levels):
            x = pyrdown(x, factor=1.2)
            feats_i = self.feature_extractor(x)
            feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode="bilinear")
            feats.append(feats_i)
        scores = self.last_conv(concatenate(feats, 1))
        return scores


class KeyNetDetector(MultiResolutionDetector):
    """Multi-scale feature detector based on KeyNet.

    This is based on the original code from paper
    "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters".
    See :cite:`KeyNet2019` for more details.

    .. image:: _static/img/keynet.jpg

    Args:
        pretrained: Download and set pretrained weights to the model.
        num_features: Number of features to detect.
        keynet_conf: Dict with initialization parameters. Do not pass it, unless you know what you are doing`.
        ori_module: for local feature orientation estimation. Default: :class:`~kornia.feature.PassLAF`,
           which does nothing. See :class:`~kornia.feature.LAFOrienter` for details.
        aff_module: for local feature affine shape estimation. Default: :class:`~kornia.feature.PassLAF`,
            which does nothing. See :class:`~kornia.feature.LAFAffineShapeEstimator` for details.
    """

    def __init__(
        self,
        pretrained: bool = False,
        num_features: int = 2048,
        keynet_conf: KeyNet_conf = keynet_default_config,
        ori_module: Optional[Module] = None,
        aff_module: Optional[Module] = None,
    ) -> None:
        model = KeyNet(pretrained, keynet_conf)
        super().__init__(model, num_features, keynet_conf["Detector_conf"], ori_module, aff_module)
