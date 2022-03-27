from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution


class _PostInitInjectionMetaClass(type):
    """To inject the ``__post_init__`` function after the creation of each instance."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class RandomGeneratorBase(nn.Module, metaclass=_PostInitInjectionMetaClass):
    """Base class for generating random augmentation parameters."""

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
        self.make_samplers(device, dtype)

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        raise NotImplementedError


class DistributionWithMapper(Distribution):
    """Wraps a distribution with a value mapper.

    This is used to restrict the output values of a given distribution by a value mapper.
    The value mapper can be functions like sigmoid, tanh, etc.

    Args:
        dist: the target distribution.
        mapper: the function or module to adjust the output from distributions.

    Example:
        >>> from torch.distributions import Normal
        >>> import torch.nn as nn
        >>> # without mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), mapper=None)
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845, -1.3986,  0.4033,  0.8380])
        >>> # with sigmoid mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), mapper=nn.Sigmoid())
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])
    """

    def __init__(self, dist: Distribution, mapper: Optional[Callable] = None) -> None:
        self.dist = dist
        self.mapper = mapper

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:  # type:ignore
        out = self.dist.rsample(sample_shape)
        if self.mapper is not None:
            out = self.mapper(out)
        return out

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:  # type:ignore
        out = self.dist.sample(sample_shape)
        if self.mapper is not None:
            out = self.mapper(out)
        return out

    def sample_n(self, n) -> torch.Tensor:
        out = self.dist.sample_n(n)
        if self.mapper is not None:
            out = self.mapper(out)
        return out

    def __getattr__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError:
            return getattr(self.dist, attr)
