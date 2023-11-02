from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli, Beta, Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _adapted_sampling, _common_param_check, _joint_range_check
from kornia.core import as_tensor, tensor, zeros
from kornia.geometry.bbox import bbox_generator
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["CutmixGenerator"]


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
            self._beta = tensor(1.0, device=device, dtype=dtype)
        else:
            self._beta = as_tensor(self.beta, device=device, dtype=dtype)
        if self.cut_size is None:
            self._cut_size = tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            self._cut_size = as_tensor(self.cut_size, device=device, dtype=dtype)

        _joint_range_check(self._cut_size, "cut_size", bounds=(0, 1))

        self.beta_sampler = Beta(self._beta, self._beta)
        self.prob_sampler = Bernoulli(tensor(float(self.p), device=device, dtype=dtype))
        self.rand_sampler = Uniform(
            tensor(0.0, device=device, dtype=dtype), tensor(1.0, device=device, dtype=dtype), validate_args=False
        )
        self.pair_sampler = Uniform(
            tensor(0.0, device=device, dtype=dtype), tensor(1.0, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")
        _device, _dtype = _extract_device_dtype([self.beta, self.cut_size])
        _common_param_check(batch_size, same_on_batch)

        if batch_size == 0:
            return {
                "mix_pairs": zeros([0, 3], device=_device, dtype=torch.long),
                "crop_src": zeros([0, 4, 2], device=_device, dtype=_dtype),
            }

        with torch.no_grad():
            batch_probs: torch.Tensor = _adapted_sampling(
                (batch_size * self.num_mix,), self.prob_sampler, same_on_batch
            )
            mix_pairs: torch.Tensor = (
                _adapted_sampling((self.num_mix, batch_size), self.pair_sampler, same_on_batch)
                .to(device=_device, dtype=_dtype)
                .argsort(dim=1)
            )

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
        x_start = _adapted_rsampling(_gen_shape, self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype) * (
            width - cut_width - 1
        )
        y_start = _adapted_rsampling(_gen_shape, self.rand_sampler, same_on_batch).to(device=_device, dtype=_dtype) * (
            height - cut_height - 1
        )
        x_start = x_start.floor()
        y_start = y_start.floor()

        crop_src = bbox_generator(x_start.squeeze(), y_start.squeeze(), cut_width, cut_height)

        # (B * num_mix, 4, 2) => (num_mix, batch_size, 4, 2)
        crop_src = crop_src.view(self.num_mix, batch_size, 4, 2)

        return {
            "mix_pairs": mix_pairs.to(device=_device, dtype=torch.long),
            "crop_src": crop_src.floor().to(device=_device, dtype=_dtype),
            "image_shape": as_tensor(batch_shape[-2:], device=_device, dtype=_dtype),
        }
