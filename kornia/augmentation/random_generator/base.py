from typing import Callable, Dict, Optional

import torch
from torch.distributions import Distribution

from kornia.core import Device, Module, Tensor


class _PostInitInjectionMetaClass(type):
    """To inject the ``__post_init__`` function after the creation of each instance."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class RandomGeneratorBase(Module, metaclass=_PostInitInjectionMetaClass):
    """Base class for generating random augmentation parameters."""

    device: Optional[Device] = None
    dtype: torch.dtype
    # If can the generator process conditional parameters according to batch_prob
    has_fit_batch_prob: bool = False

    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self) -> None:
        self.set_rng_device_and_dtype()

    def set_rng_device_and_dtype(
        self, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
    ) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        if self.device != device or self.dtype != dtype:
            self.make_samplers(device, dtype)
            self.device = device
            self.dtype = dtype

    # TODO: refine the logic with module.to()
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.set_rng_device_and_dtype(device=device, dtype=dtype)
        return self

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, Tensor]:
        raise NotImplementedError

    def fit_batch_prob(
        self, batch_shape: torch.Size, batch_prob: Tensor, params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class DistributionWithMapper(Distribution):
    """Wraps a distribution with a value mapper function.

    This is used to restrict the output values of a given distribution by a value mapper function.
    The value mapper function can be functions like sigmoid, tanh, etc.

    Args:
        dist: the target distribution.
        map_fn: the callable function to adjust the output from distributions.

    Example:
        >>> from torch.distributions import Normal
        >>> import torch.nn as nn
        >>> # without mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), map_fn=None)
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845, -1.3986,  0.4033,  0.8380])
        >>> # with sigmoid mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), map_fn=nn.Sigmoid())
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])
    """

    def __init__(self, dist: Distribution, map_fn: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self.dist = dist
        self.map_fn = map_fn

    def rsample(self, sample_shape: torch.Size) -> Tensor:  # type: ignore[override]
        out = self.dist.rsample(sample_shape)
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def sample(self, sample_shape: torch.Size) -> Tensor:  # type: ignore[override]
        out = self.dist.sample(sample_shape)
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def sample_n(self, n) -> Tensor:
        out = self.dist.sample_n(n)
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def __getattr__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError:
            return getattr(self.dist, attr)
