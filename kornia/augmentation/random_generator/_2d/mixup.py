from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli, Uniform

from kornia.augmentation.random_generator._2d.probability import random_prob_generator
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_rsampling,
    _adapted_sampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
)
from kornia.utils.helpers import _extract_device_dtype


class MixupGenerator(RandomGeneratorBase):
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.

    Returns:
        A dict of parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B,).
            - mixup_lambdas (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(self, lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None, p: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val
        self.p = p

    def __repr__(self) -> str:
        repr = f"lambda_val={self.lambda_val}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.lambda_val is None:
            lambda_val = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            lambda_val = torch.as_tensor(self.lambda_val, device=device, dtype=dtype)

        _joint_range_check(lambda_val, 'lambda_val', bounds=(0, 1))
        self.lambda_sampler = Uniform(lambda_val[0], lambda_val[1], validate_args=False)
        self.prob_sampler = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.lambda_val])

        with torch.no_grad():
            batch_probs: torch.Tensor = _adapted_sampling((batch_size,), self.prob_sampler, same_on_batch)
        mixup_pairs: torch.Tensor = torch.randperm(batch_size, device=_device, dtype=_dtype).long()
        mixup_lambdas: torch.Tensor = _adapted_rsampling((batch_size,), self.lambda_sampler, same_on_batch)
        mixup_lambdas = mixup_lambdas * batch_probs

        return dict(
            mixup_pairs=mixup_pairs.to(device=_device, dtype=torch.long),
            mixup_lambdas=mixup_lambdas.to(device=_device, dtype=_dtype),
        )


def random_mixup_generator(
    batch_size: int,
    p: float = 0.5,
    lambda_val: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        p (flot): probability of applying mixup.
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B,).
            - mixup_lambdas (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_mixup_generator(5, 0.7)
        {'mixup_pairs': tensor([4, 0, 3, 1, 2]), 'mixup_lambdas': tensor([0.6323, 0.0000, 0.4017, 0.0223, 0.1689])}
    """
    _common_param_check(batch_size, same_on_batch)
    _device, _dtype = _extract_device_dtype([lambda_val])
    lambda_val = torch.as_tensor([0.0, 1.0] if lambda_val is None else lambda_val, device=device, dtype=dtype)
    _joint_range_check(lambda_val, 'lambda_val', bounds=(0, 1))

    batch_probs: torch.Tensor = random_prob_generator(
        batch_size, p, same_on_batch=same_on_batch, device=device, dtype=dtype
    )
    mixup_pairs: torch.Tensor = torch.randperm(batch_size, device=device, dtype=dtype).long()
    mixup_lambdas: torch.Tensor = _adapted_uniform(
        (batch_size,), lambda_val[0], lambda_val[1], same_on_batch=same_on_batch
    )
    mixup_lambdas = mixup_lambdas * batch_probs

    return dict(
        mixup_pairs=mixup_pairs.to(device=_device, dtype=torch.long),
        mixup_lambdas=mixup_lambdas.to(device=_device, dtype=_dtype),
    )
