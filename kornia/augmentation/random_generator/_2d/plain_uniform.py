# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, Optional, Set, Tuple

import torch
from torch.distributions import Distribution

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.augmentation.utils.param_validation import _as_range_tensor
from kornia.augmentation.utils.random import _with_generator
from kornia.core.utils import _extract_device_dtype, _to_device_nonblocking

__all__ = ["ParameterBound", "PlainUniformGenerator"]

# factor, name, center, range
ParameterBound = Tuple[Any, str, Optional[float], Optional[Tuple[float, float]]]


class PlainUniformGenerator(RandomGeneratorBase):
    r"""Generate random parameters that distributed uniformly.

    Args:
        *samplers: a list of tuple in a pattern of ``(factor, name, center, range)``, in which
            the factor can be a two-numbered tuple, or a ``(2,)`` shaped torch torch.Tensor. The name
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
            elif isinstance(factor, torch.Tensor):
                self.register_buffer(name, factor)
            else:
                factor = _range_bound(factor, name, center=center, bounds=bound)
                self.register_buffer(name, factor)

    def __repr__(self) -> str:
        return ", ".join([f"{name}={factor}" for factor, name, _, _ in self.samplers])

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        # Built aside and assigned once: a module shared by threads may draw while another thread rebuilds it.
        sampler_dict: Dict[str, Distribution] = {}
        for factor, name, center, bound in self.samplers:
            # The attribute, not the tuple: ``torch.func.functional_call`` swaps a parameter and a parametrization moves
            # it. Numbers stay in the tuple, to be read in ``dtype``.
            if isinstance(factor, torch.nn.Parameter):
                factor = getattr(self, name)
            if center is None and bound is None:
                factor = _as_range_tensor(factor, device, dtype)
            elif center is None or bound is None:
                raise ValueError(f"`center` and `bound` should be both None or provided. Got {center} and {bound}.")
            else:
                factor = _range_bound(factor, name, center=center, bounds=bound, device=device, dtype=dtype)
            sampler_dict[name] = UniformDistribution(factor[0], factor[1], validate_args=False)
        self.sampler_dict = sampler_dict

    def _ranges_loaded(self, names: Set[str]) -> None:
        # ``make_samplers`` reads ``self.samplers``, whose numbers it needs in ``dtype`` as they were given, not rounded
        # to float32 in a buffer. A loaded range is what the generator holds from now on, in the form and placement of
        # the entry it replaces: ``forward`` returns the draws on the device and dtype of the entries, and a buffer
        # moved with the module would leave the other entries, and a later move, behind.
        held = {**self._parameters, **self._buffers}

        def loaded(factor: Any, name: str) -> Any:
            if name not in names:
                return factor
            value = held[name]
            if isinstance(factor, torch.nn.Parameter):
                return value
            if isinstance(factor, torch.Tensor):
                # The buffer itself when the module has not moved, which keeps a plain leaf that requires grad training.
                return value.to(device=factor.device, dtype=factor.dtype)
            return value.tolist()

        self.samplers = tuple(
            (loaded(factor, name), name, center, bound) for factor, name, center, bound in self.samplers
        )
        super()._ranges_loaded(names)

    @_with_generator
    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False, *, generator: torch.Generator | None = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = batch_shape[0]
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([t for t, _, _, _ in self.samplers])

        # The draws are made on the generator's device and returned on the device of the ranges. The two differ for a
        # learnable range made on CUDA in a module left on the CPU; the copy to CUDA then does not wait for the GPU.
        return {
            name: _to_device_nonblocking(
                _adapted_rsampling((batch_size,), dist, same_on_batch, generator=self._generator), _device, _dtype
            )
            for name, dist in self.sampler_dict.items()
        }
