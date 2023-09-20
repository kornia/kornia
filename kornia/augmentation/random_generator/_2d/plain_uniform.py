from typing import Any, Dict, Optional, Tuple

import torch
from torch.distributions import Distribution

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor, as_tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["ParameterBound", "PlainUniformGenerator"]

# factor, name, center, range
ParameterBound = Tuple[Any, str, Optional[float], Optional[Tuple[float, float]]]


class PlainUniformGenerator(RandomGeneratorBase):
    r"""Generate random parameters that distributed uniformly.

    Args:
        *samplers: a list of tuple in a pattern of ``(factor, name, center, range)``, in which
            the factor can be a two-numbered tuple, or a ``(2,)`` shaped torch tensor. The name
            will be the corresponding key of the returning dict. The center and range must be
            both provided worked as a validator to the given factor.

    Returns:
        A dict of parameters to be passed for transformation according the number of samplers
        and the pointed returning name of each tuple.
            - ``name``: element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    Example:
        >>> _ = torch.manual_seed(44)
        >>> PlainUniformGenerator(
        ...     ((0., 1.), "factor_1", None, None),
        ...     (torch.tensor([-0.5, 0.5]), "factor_2", 0.1, (-1., 1.)),
        ... )(torch.Size([2]))
        {'factor_1': tensor([0.7196, 0.7307]), 'factor_2': tensor([ 0.3278, -0.3657])}
    """

    def __init__(self, *samplers: ParameterBound) -> None:
        super().__init__()
        self.samplers = samplers
        names = []
        for factor, name, center, bound in samplers:
            if name in names:
                raise RuntimeError(f"factor name `{name}` has already been registered. Please check the duplication.")
            names.append(name)
            if isinstance(factor, torch.nn.Parameter):
                self.register_parameter(name, factor)
            elif isinstance(factor, Tensor):
                self.register_buffer(name, factor)
            else:
                factor = _range_bound(factor, name, center=center, bounds=bound)
                self.register_buffer(name, factor)

    def __repr__(self) -> str:
        repr = ", ".join([f"{name}={factor}" for factor, name, _, _ in self.samplers])
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.sampler_dict: Dict[str, Distribution] = {}
        for factor, name, center, bound in self.samplers:
            if center is None and bound is None:
                factor = as_tensor(factor, device=device, dtype=dtype)
            elif center is None or bound is None:
                raise ValueError(f"`center` and `bound` should be both None or provided. Got {center} and {bound}.")
            else:
                factor = _range_bound(factor, name, center=center, bounds=bound, device=device, dtype=dtype)
            self.sampler_dict.update({name: UniformDistribution(factor[0], factor[1], validate_args=False)})

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])

        return {
            name: _adapted_rsampling((batch_size,), dist, same_on_batch).to(device=_device, dtype=_dtype)
            for name, dist in self.sampler_dict.items()
        }
