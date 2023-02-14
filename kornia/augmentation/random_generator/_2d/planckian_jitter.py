from typing import Dict, List, Optional, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _joint_range_check, _range_bound
from kornia.core import Tensor, stack, tensor
from kornia.utils import get_cuda_device_if_available

__all__ = ["PlanckianJitterGenerator"]


def get_planckian_coeffs(mode: str) -> Tensor:
    default_device = get_cuda_device_if_available()
    if mode.lower() == 'blackbody':
        coefs = tensor(
            [
                [0.6743, 0.4029, 0.0013],
                [0.6281, 0.4241, 0.1665],
                [0.5919, 0.4372, 0.2513],
                [0.5623, 0.4457, 0.3154],
                [0.5376, 0.4515, 0.3672],
                [0.5163, 0.4555, 0.4103],
                [0.4979, 0.4584, 0.4468],
                [0.4816, 0.4604, 0.4782],
                [0.4672, 0.4619, 0.5053],
                [0.4542, 0.4630, 0.5289],
                [0.4426, 0.4638, 0.5497],
                [0.4320, 0.4644, 0.5681],
                [0.4223, 0.4648, 0.5844],
                [0.4135, 0.4651, 0.5990],
                [0.4054, 0.4653, 0.6121],
                [0.3980, 0.4654, 0.6239],
                [0.3911, 0.4655, 0.6346],
                [0.3847, 0.4656, 0.6444],
                [0.3787, 0.4656, 0.6532],
                [0.3732, 0.4656, 0.6613],
                [0.3680, 0.4655, 0.6688],
                [0.3632, 0.4655, 0.6756],
                [0.3586, 0.4655, 0.6820],
                [0.3544, 0.4654, 0.6878],
                [0.3503, 0.4653, 0.6933],
            ],
            device=default_device,
        )

    elif mode.upper() == 'CIED':
        coefs = tensor(
            [
                [0.5829, 0.4421, 0.2288],
                [0.5510, 0.4514, 0.2948],
                [0.5246, 0.4576, 0.3488],
                [0.5021, 0.4618, 0.3941],
                [0.4826, 0.4646, 0.4325],
                [0.4654, 0.4667, 0.4654],
                [0.4502, 0.4681, 0.4938],
                [0.4364, 0.4692, 0.5186],
                [0.4240, 0.4700, 0.5403],
                [0.4127, 0.4705, 0.5594],
                [0.4023, 0.4709, 0.5763],
                [0.3928, 0.4713, 0.5914],
                [0.3839, 0.4715, 0.6049],
                [0.3757, 0.4716, 0.6171],
                [0.3681, 0.4717, 0.6281],
                [0.3609, 0.4718, 0.6380],
                [0.3543, 0.4719, 0.6472],
                [0.3480, 0.4719, 0.6555],
                [0.3421, 0.4719, 0.6631],
                [0.3365, 0.4719, 0.6702],
                [0.3313, 0.4719, 0.6766],
                [0.3263, 0.4719, 0.6826],
                [0.3217, 0.4719, 0.6882],
            ],
            device=default_device,
        )
    else:
        raise RuntimeError(f'Unexpected mode. Gotcha {mode}')

    return stack((coefs[:, 0] / coefs[:, 1], coefs[:, 2] / coefs[:, 1]), 1)


class PlanckianJitterGenerator(RandomGeneratorBase):
    r"""Generate random planckian jitter parameters for a batch of images."""

    def __init__(self, mode: str = "blackbody", select_from: Optional[Union[int, List[int]]] = None) -> None:
        super().__init__()

        if isinstance(select_from, int):
            select_from = [select_from]

        _pl = get_planckian_coeffs(mode)
        if select_from is not None:
            self.pl = _pl[select_from]
        else:
            self.pl = _pl

        # the range of the sampling parameters
        _param_min: float = 0.0
        _param_max: float = float(self.pl.shape[0])
        self.domain = (_param_min, _param_max)
        self.mode = mode

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        idx_range = _range_bound(self.domain, 'idx_range', device=device, dtype=dtype)

        _joint_range_check(idx_range, 'idx_range', (0, self.domain[1]))
        self.pl_idx_dist = Uniform(idx_range[0], idx_range[1], validate_args=False)

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        pl_idx = _adapted_rsampling((batch_size,), self.pl_idx_dist, same_on_batch)

        return dict(coeffs=get_planckian_coeffs(self.mode)[pl_idx.long()])
