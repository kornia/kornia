from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kornia.core import Tensor


@dataclass
class DISKFeatures:
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


@dataclass
class Keypoints:
    """A temporary struct used to store keypoint detections and their log-probabilities.

    After construction, merge_with_descriptors is used to select corresponding descriptors from unet output.
    """

    xys: Tensor
    detection_logp: Tensor

    def merge_with_descriptors(self, descriptors: Tensor) -> DISKFeatures:
        """Select descriptors from a dense `descriptors` tensor, at locations given by `self.xys`"""
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return DISKFeatures(self.xys.to(torch.float32), desc, self.detection_logp)
