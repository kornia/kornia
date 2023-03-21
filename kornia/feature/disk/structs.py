from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor

@dataclass
class Features:
    kp: Tensor
    desc: Tensor
    kp_logp: Tensor

    def __post_init__(self):
        assert self.kp.device == self.desc.device
        assert self.kp.device == self.kp_logp.device

    @property
    def n(self):
        return self.kp.shape[0]

    @property
    def device(self):
        return self.kp.device

    def detached_and_grad_(self):
        return Features(
            self.kp,
            self.desc.detach().requires_grad_(),
            self.kp_logp.detach().requires_grad_(),
        )

    def requires_grad_(self, is_on):
        self.desc.requires_grad_(is_on)
        self.kp_logp.requires_grad_(is_on)

    def grad_tensors(self):
        return [self.desc, self.kp_logp]

    def to(self, *args, **kwargs):
        return Features(
            self.kp.to(*args, **kwargs),
            self.desc.to(*args, **kwargs),
            self.kp_logp.to(*args, **kwargs) if self.kp_logp is not None else None,
        )

@dataclass
class Keypoints:
    '''
    A temporary struct used to store keypoint detections and their
    log-probabilities. After construction, merge_with_descriptors is used to
    select corresponding descriptors from unet output.
    '''
    xys: Tensor
    logp: Tensor

    def merge_with_descriptors(self, descriptors: Tensor) -> Features:
        '''
        Select descriptors from a dense `descriptors` tensor, at locations
        given by `self.xys`
        '''
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return Features(self.xys.to(torch.float32), desc, self.logp)