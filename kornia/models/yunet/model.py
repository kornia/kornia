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

# based on: https://github.com/ShiqiYu/libfacedetection.train/blob/74f3aa77c63234dd954d21286e9a60703b8d0868/tasks/task1/yufacedetectnet.py  # noqa
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["YuNet"]

url: str = "https://github.com/kornia/data/raw/main/yunet_final.pth"


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

    Args:
        phase: the phase of the model, either "train" or "test".
        pretrained: if True, loads pretrained weights from the official repository.

    Example:
        >>> model = YuNet("test", pretrained=True)
        >>> img = torch.rand(1, 3, 320, 320)
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
            pretrained_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        head_data = torch.cat([o.view(o.size(0), -1) for o in head_list], 1)
        head_data = head_data.view(head_data.size(0), -1, 17)

        loc_data, conf_data, iou_data = head_data.split((14, 2, 1), dim=-1)

        if self.phase == "test":
            conf_data = torch.softmax(conf_data, dim=-1)
        else:
            loc_data = loc_data.view(loc_data.size(0), -1, 14)
            conf_data = conf_data.view(conf_data.size(0), -1, self.num_classes)
            iou_data = iou_data.view(iou_data.size(0), -1, 1)

        return {"loc": loc_data, "conf": conf_data, "iou": iou_data}
