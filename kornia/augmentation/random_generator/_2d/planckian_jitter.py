from functools import partial
from typing import Dict, List, Optional

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
    _range_bound,
)
from kornia.utils.helpers import _deprecated, _extract_device_dtype


class PlanckianJitterGenerator(RandomGeneratorBase):

    r"""Generate random color jitter parameters for a batch of images
    """

    def __init__(self, domain: List[int]) -> None:
        super().__init__()
        self.domain = domain

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:

        idx_range: torch.Tensor = _range_bound(self.domain,
                                               'idx_range',
                                               device=device, dtype=dtype)

        _joint_range_check(idx_range, 'idx_range', (0, 25))
        self.pl_idx_dist = Uniform(idx_range[0], idx_range[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool =
                False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.pl_idx_dist])
        pl_idx = _adapted_rsampling((batch_size,),
                                    self.pl_idx_dist,
                                    same_on_batch)

        return dict(idx=pl_idx.long())
