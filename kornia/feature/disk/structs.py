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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn.functional as F


@dataclass
class DISKFeatures:
    r"""A data structure holding DISK keypoints, descriptors and detection scores for an image.

    Since DISK detects a varying number of keypoints per image, `DISKFeatures` is not batched.

    Args:
        keypoints: torch.Tensor of shape :math:`(N, 2)`, where :math:`N` is the number of keypoints.
        descriptors: torch.Tensor of shape :math:`(N, D)`, where :math:`D` is the descriptor dimension.
        detection_scores: torch.Tensor of shape :math:`(N,)` where the detection score can be interpreted as
                          the log-probability of keeping a keypoint after it has been proposed (see the paper
                          section *Method → Feature distribution* for details).

    """

    keypoints: torch.Tensor
    descriptors: torch.Tensor
    detection_scores: torch.Tensor

    @property
    def n(self) -> int:
        return self.keypoints.shape[0]

    @property
    def device(self) -> Union[str, torch.device, None]:
        return self.keypoints.device

    @property
    def x(self) -> torch.Tensor:
        """Accesses the x coordinates of keypoints (along image width)."""
        return self.keypoints[:, 0]

    @property
    def y(self) -> torch.Tensor:
        """Accesses the y coordinates of keypoints (along image height)."""
        return self.keypoints[:, 1]

    def to(self, *args: Any, **kwargs: Any) -> DISKFeatures:
        """Call :func:`torch.Tensor.to` on each torch.tensor to move the keypoints, descriptors and detection scores to
        the specified device and/or data type.

        Args:
            *args: Arguments passed to :func:`torch.Tensor.to`.
            **kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

        Returns:
            A new DISKFeatures object with tensors of appropriate type and location.

        """  # noqa:D205
        return DISKFeatures(
            self.keypoints.to(*args, **kwargs),
            self.descriptors.to(*args, **kwargs),
            self.detection_scores.to(*args, **kwargs),
        )


@dataclass
class Keypoints:
    """A temporary struct used to store keypoint detections and their log-probabilities.

    After construction, merge_with_descriptors is used to select corresponding descriptors from unet output.
    """

    xys: torch.Tensor
    detection_logp: torch.Tensor

    def merge_with_descriptors(self, descriptors: torch.Tensor) -> DISKFeatures:
        """Select descriptors from a dense `descriptors` torch.Tensor, at locations given by `self.xys`."""
        dtype = descriptors.dtype
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return DISKFeatures(self.xys.to(dtype), desc, self.detection_logp)
