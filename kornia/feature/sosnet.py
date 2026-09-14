# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict

import torch
from torch import nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.download import hf_url, load_state_dict_from_url

urls: Dict[str, str | list[str]] = {}
urls["lib"] = [
    hf_url("sosnet", "sosnet_32x32_liberty.pth"),
    "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_liberty.pth",
]
urls["hp_a"] = [
    hf_url("sosnet", "sosnet_32x32_hpatches_a.pth"),
    "https://github.com/yuruntian/SOSNet/raw/master/sosnet-weights/sosnet_32x32_hpatches_a.pth",
]


class SOSNet(nn.Module):
    r"""128-dimensional SOSNet model definition for 32x32 patches.

    This is based on the original code from paper
    "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning".

    Args:
        pretrained: Download and set pretrained weights to the model.

    Shape:
        - Input: :math:`(B, 1, 32, 32)`
        - Output: :math:`(B, 128)`

    Examples:
        >>> input = torch.rand(8, 1, 32, 32)
        >>> sosnet = SOSNet()
        >>> descs = sosnet(input) # 8x128

    """

    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False),
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
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.desc_norm = nn.Sequential(nn.LocalResponseNorm(256, alpha=256.0, beta=0.5, k=0.0))
        # load pretrained model
        if pretrained:
            pretrained_dict = load_state_dict_from_url(urls["lib"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, input: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Compute SOSNet descriptors for grayscale 32x32 patches.

        Args:
            input: Input tensor with shape :math:`(B, 1, 32, 32)`.
            eps: Small constant added before normalization for stability.

        Returns:
            Descriptor tensor with shape :math:`(B, 128)`.
        """
        KORNIA_CHECK_SHAPE(input, ["B", "1", "32", "32"])
        descr = self.layers(input)
        # Half precision takes this step in float32 and casts back, for two reasons that between them
        # cover both half dtypes. (1) `desc_norm` divides by the descriptor's own L2 norm, and `eps` is
        # what keeps that division defined for a patch the network maps to exactly zero -- every
        # `Conv2d` here has `bias=False` and every `BatchNorm2d` is `affine=False`, so a constant patch
        # does. 1e-10 is not representable in float16, where it would flush to 0.0 and leave 0/0;
        # bfloat16 keeps float32's exponent range, so the guard survives there and this half of the
        # argument does not apply to it. (2) `F.local_response_norm` routes a 4-D input through
        # `avg_pool3d`, which has no CPU kernel for *either* half dtype, so the lift is what makes CPU
        # half work at all. Note the lift is therefore wider than the one `siftdesc.py` gives its own
        # 1e-10 guards, which is float16-only for reason (1) alone.
        # float32 and float64 take the original expression unchanged, bit for bit.
        if descr.dtype in (torch.float16, torch.bfloat16):
            descr = self.desc_norm(descr.float() + eps).to(descr.dtype)
        else:
            descr = self.desc_norm(descr + eps)
        descr = descr.view(descr.size(0), -1)
        return descr
