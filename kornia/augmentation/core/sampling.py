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
    def __init__(self, dist: Distribution, sampling_grads=True):
        super().__init__()
        self.dist = dist
        self.sampling_grads = sampling_grads

    def smart_sample(self, shape, same_on_batch):
        if self.sampling_grads:
            return self.rsample(shape, same_on_batch)
        else:
            return self.sample(shape, same_on_batch)

    def rsample(self, shape, same_on_batch):
        if same_on_batch:
            return self.dist.rsample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
        else:
            return self.dist.rsample(shape)

    def sample(self, shape, same_on_batch):
        if same_on_batch:
            return self.dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
        else:
            return self.dist.sample(shape)

    def train(self, mode: bool = True):
        out = super().train(mode)
        self.sampling_grads = mode
        return out


class SmartUniform(SmartSampling):
    """
    """
    def __init__(self, low: torch.Tensor, high: torch.Tensor, validate_args=False):
        dist = Uniform(low, high, validate_args=validate_args)
        super().__init__(dist)
        self.register_parameter('low', nn.Parameter(low))
        self.register_parameter('high', nn.Parameter(high))

    def to(self, *args, **kwargs):
        # Convert distribution to the corresponding device and dtype
        super().to(*args, **kwargs)
        self.dist = Uniform(self.low, self.high, validate_args=validate_args)
        return self


class SmartGaussian(SmartSampling):
    """
    """
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, validate_args=False):
        dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(dist)
        self.register_parameter('loc', nn.Parameter(loc))
        self.register_parameter('scale', nn.Parameter(scale))

    def to(self, *args, **kwargs):
        # Convert distribution to the corresponding device and dtype
        super().to(*args, **kwargs)
        self.dist = Normal(self.loc, self.scale, validate_args=validate_args)
        return self


class SmartBernoulli(SmartSampling):
    """
    """
    def __init__(self, p: torch.Tensor, temperature: float = 1e-7):
        dist = RelaxedBernoulli(temperature, p)
        super().__init__(dist)
        self.register_buffer("p", p)
        self.register_buffer("temperature", torch.as_tensor(temperature))
