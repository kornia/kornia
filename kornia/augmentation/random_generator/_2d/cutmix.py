from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli, Beta, Uniform

from kornia.augmentation.random_generator._2d.probability import random_prob_generator
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _adapted_beta,
    _adapted_rsampling,
    _adapted_sampling,
    _adapted_uniform,
    _common_param_check,
    _joint_range_check,
)
from kornia.geometry.bbox import bbox_generator
from kornia.utils.helpers import _extract_device_dtype

__all__ = [
    "CutmixGenerator"
]


class CutmixGenerator(RandomGeneratorBase):
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size (torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (num_mix, B).
            - crop_src (torch.Tensor): element-wise probabilities with a shape of (num_mix, B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        cut_size: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        beta: Optional[Union[torch.Tensor, float]] = None,
        num_mix: int = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.beta = beta
        self.num_mix = num_mix
        self.p = p

        if not (num_mix >= 1 and isinstance(num_mix, (int,))):
            raise AssertionError(f"`num_mix` must be an integer greater than 1. Got {num_mix}.")

    def __repr__(self) -> str:
        repr = f"cut_size={self.cut_size}, beta={self.beta}, num_mix={self.num_mix}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.beta is None:
            self._beta = torch.tensor(1.0, device=device, dtype=dtype)
        else:
            self._beta = torch.as_tensor(self.beta, device=device, dtype=dtype)
        if self.cut_size is None:
            self._cut_size = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            self._cut_size = torch.as_tensor(self.cut_size, device=device, dtype=dtype)

        _joint_range_check(self._cut_size, 'cut_size', bounds=(0, 1))

        self.beta_sampler = Beta(self._beta, self._beta)
        self.prob_sampler = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))
        self.rand_sampler = Uniform(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
            validate_args=False,
        )

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        if not (type(height) is int and height > 0 and type(width) is int and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
        _device, _dtype = _extract_device_dtype([self.beta, self.cut_size])
        _common_param_check(batch_size, same_on_batch)

        if batch_size == 0:
            return dict(
                mix_pairs=torch.zeros([0, 3], device=_device, dtype=torch.long),
                crop_src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype),
            )

        with torch.no_grad():
            batch_probs: torch.Tensor = _adapted_sampling(
                (batch_size * self.num_mix,), self.prob_sampler, same_on_batch
            )
        mix_pairs: torch.Tensor = torch.rand(self.num_mix, batch_size, device=_device, dtype=_dtype).argsort(dim=1)
        cutmix_betas: torch.Tensor = _adapted_rsampling((batch_size * self.num_mix,), self.beta_sampler, same_on_batch)

        # Note: torch.clamp does not accept tensor, cutmix_betas.clamp(cut_size[0], cut_size[1]) throws:
        # Argument 1 to "clamp" of "_TensorBase" has incompatible type "Tensor"; expected "float"
        cutmix_betas = torch.min(torch.max(cutmix_betas, self._cut_size[0]), self._cut_size[1])
        cutmix_rate = torch.sqrt(1.0 - cutmix_betas) * batch_probs

        cut_height = (cutmix_rate * height).floor().to(device=_device, dtype=_dtype)
        cut_width = (cutmix_rate * width).floor().to(device=_device, dtype=_dtype)
        _gen_shape = (1,)

        if same_on_batch:
            _gen_shape = (cut_height.size(0),)
            cut_height = cut_height[0]
            cut_width = cut_width[0]

        # Reserve at least 1 pixel for cropping.
        x_start: torch.Tensor = _adapted_rsampling(_gen_shape, self.rand_sampler, same_on_batch) * (
            width - cut_width - 1
        )
        y_start: torch.Tensor = _adapted_rsampling(_gen_shape, self.rand_sampler, same_on_batch) * (
            height - cut_height - 1
        )
        x_start = x_start.floor().to(device=_device, dtype=_dtype)
        y_start = y_start.floor().to(device=_device, dtype=_dtype)

        crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

        # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
        crop_src = crop_src.view(self.num_mix, batch_size, 4, 2)

        return dict(
            mix_pairs=mix_pairs.to(device=_device, dtype=torch.long),
            crop_src=crop_src.floor().to(device=_device, dtype=_dtype),
            image_shape=torch.as_tensor(batch_shape[-2:], device=_device, dtype=_dtype),
        )


def random_cutmix_generator(
    batch_size: int,
    width: int,
    height: int,
    p: float = 0.5,
    num_mix: int = 1,
    beta: Optional[torch.Tensor] = None,
    cut_size: Optional[torch.Tensor] = None,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    r"""Generate cutmix indexes and lambdas for a batch of inputs.

    Args:
        batch_size (int): the number of images. If batchsize == 1, the output will be as same as the input.
        width (int): image width.
        height (int): image height.
        p (float): probability of applying cutmix.
        num_mix (int): number of images to mix with. Default is 1.
        beta (torch.Tensor, optional): hyperparameter for generating cut size from beta distribution.
            If None, it will be set to 1.
        cut_size (torch.Tensor, optional): controlling the minimum and maximum cut ratio from [0, 1].
            If None, it will be set to [0, 1], which means no restriction.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (num_mix, B).
            - crop_src (torch.Tensor): element-wise probabilities with a shape of (num_mix, B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> random_cutmix_generator(3, 224, 224, p=0.5, num_mix=2)
        {'mix_pairs': tensor([[2, 0, 1],
                [1, 2, 0]]), 'crop_src': tensor([[[[ 35.,  25.],
                  [208.,  25.],
                  [208., 198.],
                  [ 35., 198.]],
        <BLANKLINE>
                 [[156., 137.],
                  [155., 137.],
                  [155., 136.],
                  [156., 136.]],
        <BLANKLINE>
                 [[  3.,  12.],
                  [210.,  12.],
                  [210., 219.],
                  [  3., 219.]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[ 83., 125.],
                  [177., 125.],
                  [177., 219.],
                  [ 83., 219.]],
        <BLANKLINE>
                 [[ 54.,   8.],
                  [205.,   8.],
                  [205., 159.],
                  [ 54., 159.]],
        <BLANKLINE>
                 [[ 97.,  70.],
                  [ 96.,  70.],
                  [ 96.,  69.],
                  [ 97.,  69.]]]])}
    """
    _device, _dtype = _extract_device_dtype([beta, cut_size])
    beta = torch.as_tensor(1.0 if beta is None else beta, device=device, dtype=dtype)
    cut_size = torch.as_tensor([0.0, 1.0] if cut_size is None else cut_size, device=device, dtype=dtype)
    if not (num_mix >= 1 and isinstance(num_mix, (int,))):
        raise AssertionError(f"`num_mix` must be an integer greater than 1. Got {num_mix}.")
    if not (type(height) is int and height > 0 and type(width) is int and width > 0):
        raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
    _joint_range_check(cut_size, 'cut_size', bounds=(0, 1))
    _common_param_check(batch_size, same_on_batch)

    if batch_size == 0:
        return dict(
            mix_pairs=torch.zeros([0, 3], device=_device, dtype=torch.long),
            crop_src=torch.zeros([0, 4, 2], device=_device, dtype=torch.long),
        )

    batch_probs: torch.Tensor = random_prob_generator(
        batch_size * num_mix, p, same_on_batch, device=device, dtype=dtype
    )
    mix_pairs: torch.Tensor = torch.rand(num_mix, batch_size, device=device, dtype=dtype).argsort(dim=1)
    cutmix_betas: torch.Tensor = _adapted_beta((batch_size * num_mix,), beta, beta, same_on_batch=same_on_batch)
    # Note: torch.clamp does not accept tensor, cutmix_betas.clamp(cut_size[0], cut_size[1]) throws:
    # Argument 1 to "clamp" of "_TensorBase" has incompatible type "Tensor"; expected "float"
    cutmix_betas = torch.min(torch.max(cutmix_betas, cut_size[0]), cut_size[1])
    cutmix_rate = torch.sqrt(1.0 - cutmix_betas) * batch_probs

    cut_height = (cutmix_rate * height).floor().to(device=device, dtype=_dtype)
    cut_width = (cutmix_rate * width).floor().to(device=device, dtype=_dtype)
    _gen_shape = (1,)

    if same_on_batch:
        _gen_shape = (cut_height.size(0),)
        cut_height = cut_height[0]
        cut_width = cut_width[0]

    # Reserve at least 1 pixel for cropping.
    x_start = (
        _adapted_uniform(
            _gen_shape,
            torch.zeros_like(cut_width, device=device, dtype=dtype),
            (width - cut_width - 1).to(device=device, dtype=dtype),
            same_on_batch,
        )
        .floor()
        .to(device=device, dtype=_dtype)
    )
    y_start = (
        _adapted_uniform(
            _gen_shape,
            torch.zeros_like(cut_height, device=device, dtype=dtype),
            (height - cut_height - 1).to(device=device, dtype=dtype),
            same_on_batch,
        )
        .floor()
        .to(device=device, dtype=_dtype)
    )

    crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

    # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
    crop_src = crop_src.view(num_mix, batch_size, 4, 2)

    return dict(
        mix_pairs=mix_pairs.to(device=_device, dtype=torch.long),
        crop_src=crop_src.floor().to(device=_device, dtype=_dtype),
    )
