from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import Distribution

from kornia import eye_like
from .smart_sampling import (
    SmartSampling,
    SmartBernoulli,
    SmartUniform,
)


class AugmentOperation(nn.Module):
    """
    """
    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                          List[SmartSampling]]] = None,
        mapper: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False
    ):
        super().__init__()
        if mapper is not None:
            self.mapper = self._make_mapper(mapper)
        else:
            self.mapper = None
        self.gradients_estimator = gradients_estimator
        self.same_on_batch = same_on_batch
        self.prob_dist = SmartBernoulli(p, freeze_dtype=True)
        self.prob_batch_dist = SmartBernoulli(p_batch, freeze_dtype=True)
        if sampler is not None:
            self.sampler = self._make_sampler(sampler)
        else:
            self.sampler = None

    def _make_mapper(self, mapper: Union[Callable, List[Callable]]) -> List[Callable]:
        if callable(mapper):
            return [mapper]
        if isinstance(mapper, (tuple, list)):
            return mapper
        raise ValueError

    def _make_sampler_one(self, sampler: Union[Tuple[float, float], SmartSampling]) -> Distribution:
        if isinstance(sampler, (list, tuple)):
            _sampler = SmartUniform(
                torch.tensor(sampler[0]),
                torch.tensor(sampler[1])
            )
        else:
            _sampler = sampler
        return _sampler

    def _make_sampler(
            self, sampler: Union[Tuple[float, float], List[Tuple[float, float]], List[SmartSampling],
                                 SmartSampling]
    ) -> List[Distribution]:
        """Make a list of distributions according to the parameters."""
        if (
            isinstance(sampler, (list, tuple)) and len(sampler) == 2 and
            isinstance(sampler[0], (int, float)) and isinstance(sampler[0], (int, float))
        ) or isinstance(sampler, (SmartSampling,)):
            sampler = [sampler]
        return [self._make_sampler_one(dist) for dist in sampler]

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Optional[torch.Tensor]]:
        batch_probs = self.prob_batch_dist.rsample([1], self.same_on_batch).squeeze()
        if batch_probs.bool().item():
            probs = self.prob_dist.rsample(batch_shape[:1], self.same_on_batch).squeeze()
        else:
            probs = batch_probs.expand(batch_shape[0])
        mags = None
        if self.sampler is not None:
            mags = [dist.rsample(batch_shape[:1], self.same_on_batch) for dist in self.sampler]
        if self.mapper is not None:
            mags = [mapping(mag) for mapping, mag in zip(self.mapper, mags)]
        return {"probs": probs.bool(), "magnitudes": mags}

    def apply_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(self, input: torch.Tensor, params: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, input: torch.Tensor, params: Optional[Dict[str, Optional[torch.Tensor]]] = None
    ) -> torch.Tensor:
        if params is None:
            params = self.generate_parameters(input.shape)
        if (params['probs'] == 0).all():
            return input
        if (params['probs'] == 1).all():
            return self.forwad_transform(input, params)
        inp = input[params['probs']]
        out = self.forwad_transform(inp, params)
        input[params['probs']] = out
        return input


class IntensityAugmentOperation(AugmentOperation):
    """
    """
    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                          List[SmartSampling]]] = None,
        mapper: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False
    ):
        super().__init__(
            p=p, p_batch=p_batch, sampler=sampler, mapper=mapper,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )

    def apply_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(self, input: torch.Tensor, params: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        if params['magnitudes'] is None:
            mag = None
        else:
            mag = [_mag[params['probs']] for _mag in params['magnitudes']]
            mag = mag[0] if len(mag) == 1 else mag
        if self.gradients_estimator is not None:
            with torch.no_grad():
                out = self.apply_transform(input, mag)
            out = self.gradients_estimator.apply(input, out)
        else:
            out = self.apply_transform(input, mag)
        return out


class GeometricAugmentOperation(AugmentOperation):
    """Base class for applying geometric data augmentation methods.

    Allowing apply and inverse transformations made by computed transform matrices.
    """
    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                          List[SmartSampling]]] = None,
        mapper: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
        return_transform: bool = False
    ):
        super().__init__(
            p=p, p_batch=p_batch, sampler=sampler, mapper=mapper,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )
        self.return_transform = return_transform

    def compute_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(
        self, input: torch.Tensor, transform: torch.Tensor, output_shape: torch.Size
    ) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(
        self, input: torch.Tensor, params: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if params['magnitudes'] is None:
            mag = None
        else:
            mag = [_mag[params['probs']] for _mag in params['magnitudes']]
            mag = mag[0] if len(mag) == 1 else mag
        if self.gradients_estimator is not None:
            with torch.no_grad():
                trans_mat = self.compute_transform(input, mag)
                out = self.apply_transform(input, trans_mat)
            out = self.gradients_estimator.apply(
                input, out, lambda x, shape: self.inverse_transform(x, trans_mat, shape))
        else:
            trans_mat = self.compute_transform(input, mag)
            out = self.apply_transform(input, trans_mat)
        return out, trans_mat

    def forward(
        self, input: torch.Tensor, params: Optional[Dict[str, Optional[torch.Tensor]]] = None
    ) -> torch.Tensor:
        if params is None:
            params = self.generate_parameters(input.shape)
        if (params['probs'] == 0).all():
            out, trans_mat = input, eye_like(3, input)
        elif (params['probs'] == 1).all():
            out, trans_mat = self.forwad_transform(input, params)
        else:
            inp = input[params['probs']]
            in_trans_mat = eye_like(3, input)
            _out, _trans_mat = self.forwad_transform(inp, params)
            input[params['probs']] = _out
            in_trans_mat[params['probs']] = _trans_mat
            out, trans_mat = input, in_trans_mat
        if self.return_transform:
            return out, trans_mat
        return out
