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

# based on: https://github.com/ShiqiYu/libfacedetection.train/blob/74f3aa77c63234dd954d21286e9a60703b8d0868/tasks/task1/yufacedetectnet.py
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core.download import hf_url, load_state_dict_from_url

__all__ = ["YuNet"]

_url: Union[str, List[str]] = [
    hf_url("yunet", "yunet_final.pth"),
    "https://github.com/kornia/data/raw/main/yunet_final.pth",
]


class ConvDPUnit(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1))
        self.add_module("conv2", nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels))
        if withBNRelu:
            self.add_module("bn", nn.BatchNorm2d(out_channels))
            self.add_module("relu", nn.ReLU(inplace=True))


class Conv_head(nn.Sequential):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1))
        self.add_module("bn1", nn.BatchNorm2d(mid_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv2", ConvDPUnit(mid_channels, out_channels))


class Conv4layerBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True) -> None:
        super().__init__()
        self.add_module("conv1", ConvDPUnit(in_channels, in_channels, True))
        self.add_module("conv2", ConvDPUnit(in_channels, out_channels, withBNRelu))


class YuNet(nn.Module):
    r"""YuNet face detection model.

    This is the underlying CNN model used by :py:class:`kornia.contrib.FaceDetector` for face detection.
    The model architecture is based on the method described in :cite:`facedetect-yu`.

    The input is a BGR image with pixel values in :math:`[0, 255]`, with no mean or standard deviation applied: the
    pretrained weights were trained on such images (ShiqiYu/libfacedetection.train@74f3aa7 ``src/data.py:76, 90``)
    and evaluated on images read with ``cv2.imread`` (``tasks/task1/datasets.py:53``, ``detect.py:100-106``).
    :py:class:`kornia.contrib.FaceDetector` takes RGB in :math:`[0, 255]` and flips it to BGR before this model.

    Args:
        phase: the phase of the model, either "train" or "test".
        pretrained: if True, loads pretrained weights from the official repository and returns the model in
            evaluation mode; otherwise the random weights stay in training mode.

    Example:
        >>> model = YuNet("test", pretrained=True)
        >>> img = torch.rand(1, 3, 320, 320) * 255  # BGR in [0, 255]
        >>> out = model(img)

    """

    def __init__(self, phase: str, pretrained: bool) -> None:
        super().__init__()
        self.phase = phase
        self.num_classes = 2

        self.model0 = Conv_head(3, 16, 16)
        self.model1 = Conv4layerBlock(16, 64)
        self.model2 = Conv4layerBlock(64, 64)
        self.model3 = Conv4layerBlock(64, 64)
        self.model4 = Conv4layerBlock(64, 64)
        self.model5 = Conv4layerBlock(64, 64)
        self.model6 = Conv4layerBlock(64, 64)

        self.head = nn.Sequential(
            Conv4layerBlock(64, 3 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 2 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 2 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 3 * (14 + 2 + 1), False),
        )

        if self.phase == "train":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = load_state_dict_from_url(_url, map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
            self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run YuNet face detection heads over an input image batch.

        Args:
            x: Image tensor with shape :math:`(B, C, H, W)`, where :math:`B` is
                batch size, :math:`C` is channel count, :math:`H` is height,
                and :math:`W` is width.

        Returns:
            Dictionary containing head outputs for locations, confidences, and
            landmarks at the configured detection scales.
        """
        detection_sources, head_list = [], []

        x = self.model0(x)
        x = F.max_pool2d(x, 2)
        x = self.model1(x)
        x = self.model2(x)
        x = F.max_pool2d(x, 2)
        x = self.model3(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model4(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model5(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model6(x)
        detection_sources.append(x)

        for i, h in enumerate(self.head):
            x_tmp = h(detection_sources[i])
            head_list.append(x_tmp.permute(0, 2, 3, 1).contiguous())

        head_data = torch.cat([o.flatten(1) for o in head_list], 1)
        head_data = head_data.unflatten(1, (-1, 17))

        loc_data, conf_data, iou_data = head_data.split((14, 2, 1), dim=-1)

        if self.phase == "test":
            conf_data = torch.softmax(conf_data, dim=-1)

        return {"loc": loc_data, "conf": conf_data, "iou": iou_data}
