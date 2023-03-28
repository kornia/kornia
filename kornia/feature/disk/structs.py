from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F

from kornia.core import Tensor


@dataclass
class DISKFeatures:
    r"""A data structure holding DISK keypoints, descriptors and detection scores for an image. Since DISK detects a
    varying number of keypoints per image, `DISKFeatures` is not batched.

    Args:
        keypoints: Tensor of shape :math:`(N, 2)`, where :math:`N` is the number of keypoints.
        descriptors: Tensor of shape :math:`(N, 128)`.
        detection_scores: Tensor of shape :math:`(N,)` where the detection score can be interpreted as
        the log-probability of keeping a keypoint after it has been proposed (see the paper,
        section Method->Feature distribution for details).
    """
    keypoints: Tensor
    descriptors: Tensor
    detection_scores: Tensor

    @property
    def n(self):
        return self.keypoints.shape[0]

    @property
    def device(self):
        return self.keypoints.device

    @property
    def x(self):
        return self.keypoints[:, 0]

    @property
    def y(self):
        return self.keypoints[:, 1]

    def to(self, *args, **kwargs):
        return DISKFeatures(
            self.keypoints.to(*args, **kwargs),
            self.descriptors.to(*args, **kwargs),
            self.detection_scores.to(*args, **kwargs),
        )

    def mask(self, mask) -> DISKFeatures:
        return DISKFeatures(self.keypoints[mask], self.descriptors[mask], self.detection_scores[mask])


@dataclass
class Keypoints:
    """A temporary struct used to store keypoint detections and their log-probabilities.

    After construction, merge_with_descriptors is used to select corresponding descriptors from unet output.
    """

    xys: Tensor
    detection_logp: Tensor

    def merge_with_descriptors(self, descriptors: Tensor) -> DISKFeatures:
        """Select descriptors from a dense `descriptors` tensor, at locations given by `self.xys`"""
        dtype = descriptors.dtype
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return DISKFeatures(self.xys.to(dtype), desc, self.detection_logp)
