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

# Adapted from https://github.com/Hakuyume/chainer-ssd
import math
from typing import List, Tuple

import torch

__all__ = ["PriorBox", "decode"]


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """Decode locations from predictions using priors to undo the encoding for offset regression at train time.

    Args:
        loc: location predictions for loc layers. Shape: [num_priors,4].
        priors: Prior boxes in center-offset form. Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes.

    Return:
        torch.Tensor containing decoded bounding box predictions.

    """
    boxes = torch.cat(
        (
            priors[:, 0:2] + loc[:, 0:2] * variances[0] * priors[:, 2:4],
            priors[:, 2:4] * torch.exp(loc[:, 2:4] * variances[1]),
            priors[:, 0:2] + loc[:, 4:6] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 6:8] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 8:10] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 10:12] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 12:14] * variances[0] * priors[:, 2:4],
        ),
        1,
    )
    # prepare final output
    tmp = boxes[:, 0:2] - boxes[:, 2:4] / 2
    return torch.cat((tmp, boxes[:, 2:4] + tmp, boxes[:, 4:]), dim=-1)


class PriorBox:
    """Generate prior boxes for YuNet face detection model."""

    def __init__(self, min_sizes: List[List[int]], steps: List[int], clip: bool, image_size: Tuple[int, int]) -> None:
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = image_size

        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        for i in range(4):
            if self.steps[i] != math.pow(2, (i + 3)):
                raise ValueError("steps must be [8,16,32,64]")

        self.feature_map_2th = [int(int((self.image_size[0] + 1) / 2) / 2), int(int((self.image_size[1] + 1) / 2) / 2)]
        self.feature_map_3th = [int(self.feature_map_2th[0] / 2), int(self.feature_map_2th[1] / 2)]
        self.feature_map_4th = [int(self.feature_map_3th[0] / 2), int(self.feature_map_3th[1] / 2)]
        self.feature_map_5th = [int(self.feature_map_4th[0] / 2), int(self.feature_map_4th[1] / 2)]
        self.feature_map_6th = [int(self.feature_map_5th[0] / 2), int(self.feature_map_5th[1] / 2)]

        self.feature_maps = [self.feature_map_3th, self.feature_map_4th, self.feature_map_5th, self.feature_map_6th]

    def to(self, device: torch.device, dtype: torch.dtype) -> "PriorBox":
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self) -> torch.Tensor:
        anchors: List[float] = []
        for k, f in enumerate(self.feature_maps):
            min_sizes: List[int] = self.min_sizes[k]
            # NOTE: the nested loop it's to make torchscript happy
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]

                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.tensor(anchors, device=self.device, dtype=self.dtype).view(-1, 4)
        if self.clip:
            output = output.clamp(max=1, min=0)
        return output
