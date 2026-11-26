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
    r"""Hold the integer-pixel keypoints, descriptors and detection scores that DISK detects in one image.

    ``features = disk(images)[b]`` is what :class:`DISK` returns for image ``b``: ``features.keypoints`` ``(N, 2)``,
    integer ``(x, y)`` pixels, also read as :attr:`x` and :attr:`y`, ``features.descriptors`` ``(N, D)``,
    L2-normalised, and ``features.detection_scores`` ``(N,)``, with ``N = features.n``. It is not batched, because
    DISK detects a different number of keypoints in each image; ``features.to(device)`` moves the three tensors.

    Convention:
        The keypoints, scores and descriptors that :class:`~kornia.feature.DISK` returns for one image; its
        Convention block gives their frame and order.

    Args:
        keypoints: torch.Tensor of shape ``(N, 2)``, where ``N`` is the number of keypoints.
        descriptors: torch.Tensor of shape ``(N, D)``, where ``D`` is the descriptor dimension.
        detection_scores: torch.Tensor of shape ``(N,)``, the detection heatmap value at each keypoint: a raw
                          logit of the network, not a log-probability.

    See Also:
        - :class:`DISK`: the detector, which returns one of these per image.
        - :class:`ALIKEDFeatures`: the same container for :class:`ALIKED`, whose keypoints are sub-pixel.

    """

    keypoints: torch.Tensor
    descriptors: torch.Tensor
    detection_scores: torch.Tensor

    @property
    def n(self) -> int:
        """Return the number of stored keypoints.

        Returns:
            Number of detected DISK keypoints stored in this feature container.
        """
        return self.keypoints.shape[0]

    @property
    def device(self) -> Union[str, torch.device, None]:
        """Return the device of the stored tensors.

        Returns:
            Device of the keypoints.
        """
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
        """Return a copy whose keypoints, descriptors and detection scores went through :meth:`torch.Tensor.to`.

        Args:
            *args: Arguments passed to :meth:`torch.Tensor.to`.
            **kwargs: Keyword arguments passed to :meth:`torch.Tensor.to`.

        Returns:
            A new DISKFeatures object with tensors of appropriate type and location.

        """
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

        # The integer pixels go to at least float32: bfloat16 holds no odd integer above 256, float16 none above 2048.
        xys = self.xys.to(torch.promote_types(dtype, torch.float32))
        return DISKFeatures(xys, desc, self.detection_logp.to(dtype))
