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
from kornia.feature._layout import _mark_patch_convolutions, _patch_conv_layout

urls: Dict[str, str | list[str]] = {}
urls["liberty"] = [
    hf_url("tfeat", "tfeat-liberty.params"),
    "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-liberty.params",
]  # pylint: disable
urls["notredame"] = [
    hf_url("tfeat", "tfeat-notredame.params"),
    "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-notredame.params",
]  # pylint: disable
urls["yosemite"] = [
    hf_url("tfeat", "tfeat-yosemite.params"),
    "https://github.com/vbalnt/tfeat/raw/master/pretrained-models/tfeat-yosemite.params",
]  # pylint: disable


class TFeat(nn.Module):
    r"""Describe 32 x 32 grayscale patches with the shallow TFeat CNN, without L2 normalisation.

    ``tfeat = TFeat()`` builds the network with its pretrained weights (trained on Liberty), in eval mode, and
    ``descs = tfeat(patches)`` takes grayscale patches ``(B, 1, 32, 32)`` and returns ``(B, 128)`` descriptors with
    entries in ``[-1, 1]``, not unit vectors (see the Convention block). It has two convolutions and one linear layer.
    For unit-norm descriptors of the same patches, use :class:`HardNet` instead; to describe LAFs on an image, pass it
    to :class:`LAFDescriptor`. This is based on the original code from paper "Learning local feature descriptors
    with triplets and shallow convolutional neural networks". See :cite:`TFeat2016` for more details.

    Convention:
        - It takes the patches of :class:`~kornia.feature.HardNet`, with the intensity rule of HardNet's Convention
          block, but returns the final ``tanh`` layer of the reference implementation without L2 normalisation: each
          entry lies in ``[-1, 1]`` and the rows are not unit vectors. Normalise them where a distance threshold
          assumes unit descriptors.

    Args:
        pretrained: Download and set pretrained weights to the model.

    Returns:
        torch.Tensor: TFeat descriptor of the patches.

    Shape:
        - Input: ``(B, 1, 32, 32)``
        - Output: ``(B, 128)``

    See Also:
        - :class:`HardNet`: a deeper network for the same patches, with unit-norm descriptors.
        - :class:`SOSNet`: a deeper network for the same patches that also starts with an ``InstanceNorm2d``.
        - :class:`LAFDescriptor`: runs a patch descriptor such as this one on a patch per LAF of an image.

    Examples:
        >>> input = torch.rand(16, 1, 32, 32)
        >>> tfeat = TFeat(pretrained=False).eval()
        >>> descs = tfeat(input) # 16x128

    """

    patch_size = 32

    def __init__(self, pretrained: bool = True) -> None:
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
            pretrained_dict = load_state_dict_from_url(urls["liberty"], map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
            self.eval()

    def __prepare_scriptable__(self) -> "TFeat":
        _mark_patch_convolutions(self.features)
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute TFeat descriptors for grayscale 32x32 patches.

        Args:
            input: Input tensor with shape ``(B, 1, 32, 32)``.

        Returns:
            Descriptor tensor with shape ``(B, 128)``.
        """
        KORNIA_CHECK_SHAPE(input, ["B", "1", "32", "32"])
        x = input
        for layer in self.features:
            # The leading InstanceNorm2d returns the default layout, so the layout is set per convolution; a
            # convolution's output already has it, which makes the second call free. A scripted loop also needs the
            # mark (see _mark_patch_convolutions). Ruff's SIM114 would merge the two tests with ``or``, which
            # TorchScript rejects.
            if isinstance(layer, nn.Conv2d):  # noqa: SIM114
                x = _patch_conv_layout(x)
            elif getattr(layer, "_patch_conv_input", False):
                x = _patch_conv_layout(x)
            x = layer(x)
        x = x.flatten(1)
        return self.descr(x)
