from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar

import torch
from torch import nn
from torch.autograd import Function
from torch.distributions import Bernoulli, RelaxedBernoulli

from kornia.augmentation.base import _AugmentationBase
from kornia.core import Module, Tensor

T = TypeVar("T", bound="OperationBase")


class OperationBase(Module):
    """Base class of differentiable augmentation operations.

    Args:
        operation: Kornia augmentation module.
        initial_magnitude: targeted magnitude parameter name and its initial magnitude value.
            The magnitude parameter name shall align with the attribute inside the random_generator
            in each augmentation. If None, the augmentation will be randomly applied according to
            the augmentation sampling range.
        temperature: temperature for RelaxedBernoulli distribution used during training.
        is_batch_operation: determine if to obtain the probability from `p` or `p_batch`.
            Set to True for most non-shape-persistent operations (e.g. cropping).
    """

    def __init__(
        self,
        operation: _AugmentationBase,
        initial_magnitude: Optional[List[Tuple[str, Optional[float]]]] = None,
        temperature: float = 0.1,
        is_batch_operation: bool = False,
        magnitude_fn: Optional[Callable[[Tensor], Tensor]] = None,
        gradient_estimator: Optional[Type[Function]] = None,
        symmetric_megnitude: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(operation, _AugmentationBase):
            raise ValueError(f"Only Kornia augmentations supported. Got {operation}.")

        self.op = operation

        self._init_magnitude(initial_magnitude)

        # Avoid skipping the sampling in `__batch_prob_generator__`
        self.probability_range = (1e-7, 1 - 1e-7)
        self._is_batch_operation = is_batch_operation
        if is_batch_operation:
            self._probability = nn.Parameter(torch.empty(1).fill_(self.op.p_batch))
        else:
            self._probability = nn.Parameter(torch.empty(1).fill_(self.op.p))

        if temperature < 0:
            raise ValueError(f"Expect temperature value greater than 0. Got {temperature}.")
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

        self.symmetric_megnitude = symmetric_megnitude
        self._magnitude_fn = self._init_magnitude_fn(magnitude_fn)
        self._gradient_estimator = gradient_estimator

    def _init_magnitude_fn(self, magnitude_fn: Optional[Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
        def _identity(x: Tensor) -> Tensor:
            return x

        def _random_flip(fn: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
            def f(x: Tensor) -> Tensor:
                flip = torch.rand((x.shape[0],), device=x.device) > 0.5
                return fn(x) * flip

            return f

        if magnitude_fn is None:
            magnitude_fn = _identity

        if self.symmetric_megnitude:
            return _random_flip(magnitude_fn)

        return magnitude_fn

    def _init_magnitude(self, initial_magnitude: Optional[List[Tuple[str, Optional[float]]]]) -> None:
        if isinstance(initial_magnitude, (list, tuple)):
            if not all(isinstance(ini_mag, (list, tuple)) and len(ini_mag) == 2 for ini_mag in initial_magnitude):
                raise ValueError(f"`initial_magnitude` shall be a list of 2-element tuples. Got {initial_magnitude}")
            if len(initial_magnitude) != 1:
                raise NotImplementedError("Multi magnitudes operations are not yet supported.")

        if initial_magnitude is None:
            self._factor_name = None
            self._magnitude = None
            self.magnitude_range = None
        else:
            self._factor_name = initial_magnitude[0][0]
            if self.op._param_generator is not None:
                self.magnitude_range = getattr(self.op._param_generator, self._factor_name)
            else:
                raise ValueError(f"No valid magnitude `{self._factor_name}` found in `{self.op._param_generator}`.")

            self._magnitude = None
            if initial_magnitude[0][1] is not None:
                self._magnitude = nn.Parameter(torch.empty(1).fill_(initial_magnitude[0][1]))

    def _update_probability_gen(self, relaxation: bool) -> None:
        if relaxation:
            if self._is_batch_operation:
                self.op._p_batch_gen = RelaxedBernoulli(self.temperature, self.probability)
            else:
                self.op._p_gen = RelaxedBernoulli(self.temperature, self.probability)
        elif self._is_batch_operation:
            self.op._p_batch_gen = Bernoulli(self.probability)
        else:
            self.op._p_gen = Bernoulli(self.probability)

    def train(self: T, mode: bool = True) -> T:
        self._update_probability_gen(relaxation=mode)

        return super().train(mode=mode)

    def eval(self: T) -> T:
        return self.train(False)

    def forward_parameters(self, batch_shape: torch.Size, mag: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if mag is None:
            mag = self.magnitude
        # Need to setup the sampler again for each update.
        # Otherwise, an error for updating the same graph twice will be thrown.
        self._update_probability_gen(relaxation=True)
        params = self.op.forward_parameters(batch_shape)

        if mag is not None:
            if self._factor_name is None:
                raise RuntimeError("No factor found in the params while `mag` is provided.")
            # For single factor operations, this is equivalent to `same_on_batch=True`
            params[self._factor_name] = params[self._factor_name].zero_() + mag

        if self._factor_name is not None:
            params[self._factor_name] = self._magnitude_fn(params[self._factor_name])

        return params

    def forward(self, input: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        if params is None:
            params = self.forward_parameters(input.shape)

        batch_prob = params["batch_prob"][(...,) + ((None,) * (len(input.shape) - 1))].to(device=input.device)

        if self._gradient_estimator is not None:
            # skip the gradient computation if gradient estimator is provided.
            with torch.no_grad():
                output = self.op(input, params=params)
            output = batch_prob * output + (1 - batch_prob) * input
            if self.magnitude is None:
                # If magnitude is None, make the grad w.r.t the input
                return self._gradient_estimator.apply(input, output)
            # If magnitude is not None, make the grad w.r.t the magnitude
            return self._gradient_estimator.apply(self.magnitude, output)
        return batch_prob * self.op(input, params=params) + (1 - batch_prob) * input

    @property
    def transform_matrix(self) -> Optional[Tensor]:
        if hasattr(self.op, "transform_matrix"):
            return self.op.transform_matrix
        return None

    @property
    def magnitude(self) -> Optional[Tensor]:
        if self._magnitude is None:
            return None
        mag = self._magnitude
        if self.magnitude_range is not None:
            return mag.clamp(*self.magnitude_range)
        return mag

    @property
    def probability(self) -> Tensor:
        p = self._probability.clamp(*self.probability_range)
        return p
