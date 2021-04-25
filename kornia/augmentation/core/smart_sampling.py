from typing import Optional

import torch
import torch.nn as nn

from torch.distributions import (
    Distribution,
    RelaxedBernoulli,
    Bernoulli,
    Uniform,
    Normal
)


class SmartSampling(nn.Module):
    """
    Args:
        freeze_dtype (bool): if True, keep the dtype unchanged from  `half()`, `float()`, `double()`.
            if False, it will perform like a normal nn.Module. It is especially useful for some distributions
            that contain unsupoorted dtypes.
    """
    def __init__(
        self, validate_args: Optional[bool] = None, sampling_grads: bool = True, freeze_dtype: bool = False
    ):
        super().__init__()
        self.validate_args = validate_args
        self.sampling_grads = sampling_grads
        self.freeze_dtype = freeze_dtype

    def smart_sample(self, shape, same_on_batch: bool = False):
        if self.sampling_grads:
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
        self.sampling_grads = mode
        return out

    def reconstruct_sampler(self) -> Distribution:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled.
        """
        raise NotImplementedError

    def _apply(self, fn):
        out = super()._apply(fn)
        self.dist = self.reconstruct_sampler()
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

    def bfloat16(sel):
        if not self.freeze_dtype:
            return super().bfloat16()
        return self

    def to(self, *args, **kwargs):
        # TODO: ungly implementation here.
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            if convert_to_format is not None and t.dim() == 4:
                return t.to(device, dtype if t.is_floating_point() and not self.freeze_dtype else None, non_blocking,
                            memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() and not self.freeze_dtype else None, non_blocking)

        return self._apply(convert)


class SmartUniform(SmartSampling):
    """
    Example:
        >>> s_dist = SmartUniform(torch.tensor(0.), torch.tensor(1.))
        >>> s_dist.double().smart_sample((10,)).dtype
        torch.float64
        >>> s_dist.float().smart_sample((10,)).dtype
        torch.float32
        >>> s_dist.half().smart_sample((10,)).dtype
        torch.float16
        >>> s_dist = SmartUniform(torch.tensor(0.), torch.tensor(1.), freeze_dtype=True)
        >>> s_dist.double().smart_sample((10,)).dtype
        torch.float32
        >>> s_dist.to(torch.tensor(0., dtype=torch.float64)).smart_sample((10,)).dtype
        torch.float32
    """
    def __init__(
        self, low: torch.Tensor, high: torch.Tensor, validate_args: Optional[bool] = None,
        freeze_dtype: bool = False
    ):
        super().__init__(validate_args, freeze_dtype=freeze_dtype)
        self.register_parameter('low', nn.Parameter(low))
        self.register_parameter('high', nn.Parameter(high))
        self.dist = self.reconstruct_sampler()

    def reconstruct_sampler(self) -> None:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled.
        """
        return Uniform(self.low, self.high, validate_args=self.validate_args)


class SmartGaussian(SmartSampling):
    """
    Example:
        >>> s_dist = SmartGaussian(torch.tensor(0.), torch.tensor(1.))
        >>> s_dist.double().smart_sample((10,)).dtype
        torch.float64
        >>> s_dist.float().smart_sample((10,)).dtype
        torch.float32
        >>> s_dist.half().smart_sample((10,)).dtype
        torch.float16
    """
    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, validate_args: Optional[bool] = None,
        freeze_dtype: bool = False
    ):
        super().__init__(validate_args, freeze_dtype=freeze_dtype)
        self.register_parameter('loc', nn.Parameter(loc))
        self.register_parameter('scale', nn.Parameter(scale))
        self.dist = self.reconstruct_sampler()

    def reconstruct_sampler(self) -> None:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled.
        """
        return Normal(self.loc, self.scale, validate_args=self.validate_args)


class SmartBernoulli(SmartSampling):
    """
    Example:
        >>> s_dist = SmartBernoulli(torch.tensor(0.5))
        >>> s_dist.double().smart_sample((10,)).dtype
        torch.float64
        >>> s_dist.float().smart_sample((10,)).dtype
        torch.float32
    """
    def __init__(
        self, p: torch.Tensor, temperature: float = 1e-7, validate_args: Optional[bool] = None,
        freeze_dtype: bool = True
    ):
        # dtype is frozen to avoid `RuntimeError: "clamp_cpu" not implemented for 'Half'`.
        super().__init__(validate_args, freeze_dtype=freeze_dtype)
        self.register_buffer("p", p)
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.dist = self.reconstruct_sampler()

    def reconstruct_sampler(self) -> None:
        """When .cuda(), .cpu(), .double() is called, the sampler will need to be resampled.
        """
        return RelaxedBernoulli(self.temperature, self.p, validate_args=self.validate_args)
