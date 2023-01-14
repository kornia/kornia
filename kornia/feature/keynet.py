import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from kornia.core import Module, Tensor, concatenate, tensor, where, zeros
from kornia.filters import SpatialGradient
from kornia.geometry.subpix import NonMaximaSuppression2d
from kornia.geometry.transform import pyrdown
from kornia.utils.helpers import map_location_to_cpu

from .laf import laf_from_center_scale_ori
from .orientation import PassLAF


class KeyNet_conf(TypedDict):
    num_filters: int
    num_levels: int
    kernel_size: int
    nms_size: int
    pyramid_levels: int
    up_levels: int
    scale_factor_levels: float
    s_mult: float


keynet_default_config: KeyNet_conf = {
    # Key.Net Model
    'num_filters': 8,
    'num_levels': 3,
    'kernel_size': 5,
    # Extraction Parameters
    'nms_size': 15,
    'pyramid_levels': 4,
    'up_levels': 1,
    'scale_factor_levels': math.sqrt(2),
    's_mult': 22.0,
}

KeyNet_URL = "https://github.com/axelBarroso/Key.Net-Pytorch/raw/main/model/weights/keynet_pytorch.pth"


class _FeatureExtractor(Module):
    """Helper class for KeyNet.

    It loads both, the handcrafted and learnable blocks
    """

    def __init__(self):
        super().__init__()

        self.hc_block = _HandcraftedBlock()
        self.lb_block = _LearnableBlock()

    def forward(self, x: Tensor) -> Tensor:
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb


class _HandcraftedBlock(Module):
    """Helper class for KeyNet, it defines the handcrafted filters within the Key.Net handcrafted block."""

    def __init__(self):
        super().__init__()
        self.spatial_gradient = SpatialGradient('sobel', 1)

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

    def __init__(self, in_channels: int = 10):
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
):
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

    Args:
        pretrained: Download and set pretrained weights to the model.
        keynet_conf: Dict with initiliazation parameters. Do not pass it, unless you know what you are doing`.

    Returns:
        KeyNet response score.

    Shape:
        - Input: :math:`(B, 1, H, W)`
        - Output: :math:`(B, 1, H, W)`
    """

    def __init__(self, pretrained: bool = False, keynet_conf: KeyNet_conf = keynet_default_config):
        super().__init__()

        num_filters = keynet_conf['num_filters']
        self.num_levels = keynet_conf['num_levels']
        kernel_size = keynet_conf['kernel_size']
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
            pretrained_dict = torch.hub.load_state_dict_from_url(KeyNet_URL, map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict['state_dict'], strict=True)
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
            feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear')
            feats.append(feats_i)
        scores = self.last_conv(concatenate(feats, 1))
        return scores


class KeyNetDetector(Module):
    """Multi-scale feature detector based on KeyNet.

    This is based on the original code from paper
    "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters".
    See :cite:`KeyNet2019` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model.
        num_features: Number of features to detect.
        keynet_conf: Dict with initiliazation parameters. Do not pass it, unless you know what you are doing`.
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
    ):
        super().__init__()
        self.model = KeyNet(pretrained, keynet_conf)
        # Load extraction configuration
        self.num_pyramid_levels = keynet_conf['pyramid_levels']
        self.num_upscale_levels = keynet_conf['up_levels']
        self.scale_factor_levels = keynet_conf['scale_factor_levels']
        self.mr_size = keynet_conf['s_mult']
        self.nms_size = keynet_conf['nms_size']
        self.nms = NonMaximaSuppression2d((self.nms_size, self.nms_size))
        self.num_features = num_features

        if ori_module is None:
            self.ori: Module = PassLAF()
        else:
            self.ori = ori_module

        if aff_module is None:
            self.aff: Module = PassLAF()
        else:
            self.aff = aff_module

    def remove_borders(self, score_map, borders: int = 15):
        """It removes the borders of the image to avoid detections on the corners."""
        mask = torch.zeros_like(score_map)
        mask[:, :, borders:-borders, borders:-borders] = 1
        return mask * score_map

    def detect_features_on_single_level(
        self, level_img: Tensor, num_kp: int, factor: Tuple[float, float]
    ) -> Tuple[Tensor, Tensor]:
        det_map = self.nms(self.remove_borders(self.model(level_img)))
        device = level_img.device
        dtype = level_img.dtype
        yx = det_map.nonzero()[:, 2:].t()
        scores = det_map[0, 0, yx[0], yx[1]]  # keynet supports only non-batched images

        scores_sorted, indices = torch.sort(scores, descending=True)

        indices = indices[where(scores_sorted > 0.0)]
        yx = yx[:, indices[:num_kp]].t()
        current_kp_num = len(yx)
        xy_projected = yx.view(1, current_kp_num, 2).flip(2) * tensor(factor, device=device, dtype=dtype)
        scale_factor = 0.5 * (factor[0] + factor[1])
        scale = scale_factor * self.mr_size * torch.ones(1, current_kp_num, 1, 1, device=device, dtype=dtype)
        lafs = laf_from_center_scale_ori(xy_projected, scale, zeros(1, current_kp_num, 1, device=device, dtype=dtype))
        return scores_sorted[:num_kp], lafs

    def detect(self, img: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # Compute points per level
        num_features_per_level: List[float] = []
        tmp = 0.0
        factor_points = self.scale_factor_levels**2
        levels = self.num_pyramid_levels + self.num_upscale_levels + 1
        for idx_level in range(levels):
            tmp += factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            nf = self.num_features * factor_points ** (-1 * (idx_level - self.num_upscale_levels))
            num_features_per_level.append(nf)
        num_features_per_level = list(map(lambda x: int(x / tmp), num_features_per_level))

        _, _, h, w = img.shape
        img_up = img
        cur_img = img
        all_responses: List[Tensor] = []
        all_lafs: List[Tensor] = []
        # Extract features from the upper levels
        for idx_level in range(self.num_upscale_levels):
            nf = num_features_per_level[len(num_features_per_level) - self.num_pyramid_levels - 1 - (idx_level + 1)]
            num_points_level = int(nf)

            # Resize input image
            up_factor = self.scale_factor_levels ** (1 + idx_level)
            nh, nw = int(h * up_factor), int(w * up_factor)
            up_factor_kpts = (float(w) / float(nw), float(h) / float(nh))
            img_up = F.interpolate(img_up, (nh, nw), mode='bilinear', align_corners=False)

            cur_scores, cur_lafs = self.detect_features_on_single_level(img_up, num_points_level, up_factor_kpts)

            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)

        # Extract features from the downsampling pyramid
        for idx_level in range(self.num_pyramid_levels + 1):
            if idx_level > 0:
                cur_img = pyrdown(cur_img, factor=self.scale_factor_levels)
                _, _, nh, nw = cur_img.shape
                factor = (float(w) / float(nw), float(h) / float(nh))
            else:
                factor = (1.0, 1.0)

            num_points_level = int(num_features_per_level[idx_level])
            if idx_level > 0 or (self.num_upscale_levels > 0):
                nf2 = [num_features_per_level[a] for a in range(0, idx_level + 1 + self.num_upscale_levels)]
                res_points = Tensor(nf2).sum().item()
                num_points_level = int(res_points)

            cur_scores, cur_lafs = self.detect_features_on_single_level(cur_img, num_points_level, factor)
            all_responses.append(cur_scores.view(1, -1))
            all_lafs.append(cur_lafs)
        responses: Tensor = concatenate(all_responses, 1)
        lafs: Tensor = concatenate(all_lafs, 1)
        if lafs.shape[1] > self.num_features:
            responses, idxs = torch.topk(responses, k=self.num_features, dim=1)
            lafs = torch.gather(lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3))
        return responses, lafs

    def forward(self, img: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Three stage local feature detection. First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image to extract features with shape [1xCxHxW]. KeyNetDetector does not support batch processing,
        because the number of detections is different on each image.
            mask: a mask with weights where to apply the response function. The shape must be the same as
              the input image.

        Returns:
            lafs: shape [1xNx2x3]. Detected local affine frames.
            responses: shape [1xNx1]. Response function values for corresponding lafs
        """
        if img.shape[0] != 1:
            raise ValueError("KeyNet supports only single-image input")
        responses, lafs = self.detect(img, mask)
        lafs = self.aff(lafs, img)
        lafs = self.ori(lafs, img)
        return lafs, responses
