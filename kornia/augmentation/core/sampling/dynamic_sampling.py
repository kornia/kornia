from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal, RelaxedBernoulli, Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import Transform

__all__ = ["DynamicSampling", "DynamicUniform", "DynamicGaussian", "DynamicBernoulli"]


class DynamicSampling(nn.Module):
    """Base class for dynamic sampling from distributions.

    Args:
        if_rsample (bool): if to use reparametrized sample or not.
        freeze_dtype (bool): if True, keep the dtype unchanged from  `half()`, `float()`, `double()`.
            if False, it will perform like a normal nn.Module. It is especially useful for some distributions
            that contain unsupoorted dtypes.
    """

    def __init__(
        self,
        transform: Optional[Transform] = None,
        validate_args: Optional[bool] = None,
        if_rsample: bool = True,
        freeze_dtype: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.validate_args = validate_args
        self.if_rsample = if_rsample
        self.freeze_dtype = freeze_dtype

    def dynamic_register(self, name: str, param: Any) -> None:
        if isinstance(param, nn.Parameter):
            self.register_parameter(name, param)
        else:
            self.register_buffer(name, torch.as_tensor(param))

    def dynamic_sample(self, shape, same_on_batch: bool = False):
        if self.if_rsample:
            return self.rsample(shape, same_on_batch)
        else:
            return self.sample(shape, same_on_batch)

    def rsample(self, shape, same_on_batch: bool = False):
        if same_on_batch:
            return self.dist.rsample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
        else:
            return self.dist.rsample(shape)

    def sample(self, shape, same_on_batch: bool = False):
        if same_on_batch:
            return self.dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
        else:
            return self.dist.sample(shape)

    def train(self, mode: bool = True):
        out = super().train(mode)
        self.if_rsample = mode
        return out

    def construct(self) -> None:
        dist = self.construct_sampler()
        transform: Transform
        if self.transform is None:
            transform = torch.distributions.transforms.ComposeTransform([])
        else:
            transform = self.transform
        self.dist = TransformedDistribution(dist, transform)

    def construct_sampler(self) -> Distribution:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled."""
        raise NotImplementedError

    def _apply(self, fn):
        out = super()._apply(fn)
        self.dist = self.construct_sampler()
        return out

    def float(self):
        if not self.freeze_dtype:
            return super().float()
        return self

    def double(self):
        if not self.freeze_dtype:
            return super().double()
        return self

    def half(self):
        if not self.freeze_dtype:
            return super().half()
        return self

    def bfloat16(self):
        if not self.freeze_dtype:
            return super().bfloat16()
        return self

    def to(self, *args, **kwargs):
        # TODO: ungly implementation here.
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError(
                    'nn.Module.to only accepts floating point ' 'dtypes, but got desired dtype={}'.format(dtype)
                )

        def convert(t):
            if convert_to_format is not None and t.dim() == 4:
                return t.to(
                    device,
                    dtype if t.is_floating_point() and not self.freeze_dtype else None,
                    non_blocking,
                    memory_format=convert_to_format,
                )
            return t.to(device, dtype if t.is_floating_point() and not self.freeze_dtype else None, non_blocking)

        return self._apply(convert)

    def entropy(self):
        return self.dist.entropy()


class DynamicUniform(DynamicSampling):
    """For dynamic sampling from uniform distributions.

    Example:
        >>> s_dist = DynamicUniform(torch.tensor(0.), torch.tensor(1.))
        >>> s_dist.double().dynamic_sample((10,)).dtype
        torch.float64
        >>> s_dist.float().dynamic_sample((10,)).dtype
        torch.float32
        >>> s_dist.half().dynamic_sample((10,)).dtype
        torch.float16
        >>> s_dist = DynamicUniform(torch.tensor(0.), torch.tensor(1.), freeze_dtype=True)
        >>> s_dist.double().dynamic_sample((10,)).dtype
        torch.float32
        >>> s_dist.to(torch.tensor(0., dtype=torch.float64)).dynamic_sample((10,)).dtype
        torch.float32
    """

    def __init__(
        self,
        low: Union[torch.Tensor, float],
        high: Union[torch.Tensor, float],
        transform: Optional[Transform] = None,
        validate_args: Optional[bool] = None,
        freeze_dtype: bool = False,
    ):
        super().__init__(transform, validate_args, freeze_dtype=freeze_dtype)
        self.dynamic_register('low', low)
        self.dynamic_register('high', high)
        self.construct()

    def construct_sampler(self) -> Distribution:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled."""
        return Uniform(self.low, self.high, validate_args=self.validate_args)


class DynamicGaussian(DynamicSampling):
    """For dynamic sampling from guassian distributions.

    Note:
        Whilst augmentating an image, we normally want the scale of a Guassian Distritubion to be
        as large as possible to keep the maximum sampling range. This can be reflected as to make
        the second order derivative as low as possilbe, or to maximize the distribution entropy.

    Example:
        >>> s_dist = DynamicGaussian(torch.tensor(0.), torch.tensor(1.))
        >>> s_dist.double().dynamic_sample((10,)).dtype
        torch.float64
        >>> s_dist.float().dynamic_sample((10,)).dtype
        torch.float32
        >>> s_dist.half().dynamic_sample((10,)).dtype
        torch.float16
    """

    def __init__(
        self,
        loc: Union[torch.Tensor, float],
        scale: Union[torch.Tensor, float],
        transform: Optional[Transform] = None,
        validate_args: Optional[bool] = None,
        eps: float = 1e-7,
        freeze_dtype: bool = False,
    ):
        super().__init__(transform, validate_args, freeze_dtype=freeze_dtype)
        self.dynamic_register('_loc', loc)
        self.dynamic_register('_scale', scale)
        self.eps = eps
        self.construct()

    def construct_sampler(self) -> Distribution:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled."""
        return Normal(
            self._loc,
            torch.maximum(self._scale, self._scale * 0.0 + self.eps),  # type: ignore
            validate_args=self.validate_args,
        )


class DynamicBernoulli(DynamicSampling):
    """For dynamic sampling from Bernoulli distributions.

    Example:
        >>> s_dist = DynamicBernoulli(torch.tensor(0.5), freeze_dtype=True)
        >>> s_dist.double().dynamic_sample((10,)).dtype
        torch.float32
        >>> s_dist.float().dynamic_sample((10,)).dtype
        torch.float32
        >>> s_dist = DynamicBernoulli(torch.tensor(0.5), freeze_dtype=False)
        >>> s_dist.double().dynamic_sample((10,)).dtype
        torch.float64
    """

    def __init__(
        self,
        p: torch.Tensor,
        temperature: float = 1e-7,
        transform: Optional[Transform] = None,
        validate_args: Optional[bool] = None,
        freeze_dtype: bool = True,
    ):
        # dtype is frozen to avoid `RuntimeError: "clamp_cpu" not implemented for 'Half'`.
        super().__init__(transform, validate_args, freeze_dtype=freeze_dtype)
        self.dynamic_register("p", p)
        self.dynamic_register("temperature", temperature)
        self.construct()

    def construct_sampler(self) -> Distribution:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled."""
        return RelaxedBernoulli(self.temperature, self.p, validate_args=self.validate_args)
