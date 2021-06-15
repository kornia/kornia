from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import Distribution

from kornia.augmentation.core.sampling import DynamicBernoulli, DynamicSampling, DynamicUniform
from kornia.geometry.epipolar.numeric import eye_like

Parameters = NamedTuple("Parameters", [('probs', torch.Tensor), ('magnitudes', List[torch.Tensor])])


class AugmentOperation(nn.Module):
    """ """

    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = None,
        mapper: Optional[List[Callable]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
    ):
        super().__init__()
        self.gradients_estimator = gradients_estimator
        self.same_on_batch = same_on_batch
        self.prob_dist = DynamicBernoulli(p, freeze_dtype=True)
        self.prob_batch_dist = DynamicBernoulli(p_batch, freeze_dtype=True)
        self.sampler = self._make_sampler(sampler) if sampler is not None else None
        self.mapper = mapper

    def _make_sampler_one(self, sampler: Union[Tuple[float, float], DynamicSampling]) -> DynamicSampling:
        _sampler: DynamicSampling
        if isinstance(sampler, (list, tuple)):
            _sampler = DynamicUniform(*sampler)
        else:
            _sampler = sampler
        return _sampler

    def _make_sampler(self, sampler: List[Union[Tuple[float, float], DynamicSampling]]) -> nn.ModuleList:
        """Make a list of distributions according to the parameters."""
        return nn.ModuleList([self._make_sampler_one(dist) for dist in sampler])

    def distribution_entropy(self, reduce: Optional[str] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.sampler is None:
            raise ValueError(f"No sampler found.")

        dists = [dist.entropy() for dist in self.sampler]
        if reduce is None:
            return dists
        if reduce == "sum":
            return torch.stack(dists).sum(dim=0).squeeze()
        if reduce == "mean":
            return torch.stack(dists).mean(dim=0).squeeze()
        raise NotImplementedError(f"Not implemented `reduce`: {reduce}.")

    def get_batch_probabilities(self, input: torch.Tensor) -> torch.Tensor:
        """Generate batch probabilites."""
        batch_shape = input.shape
        batch_probs = self.prob_batch_dist.rsample([1], self.same_on_batch).squeeze()
        if batch_probs.bool().item():
            probs = self.prob_dist.rsample(batch_shape[:1], self.same_on_batch).squeeze()
        else:
            probs = batch_probs.expand(batch_shape[0])
        return probs

    def get_param_magnitudes(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Parameter sampling methods."""
        batch_shape = input.shape
        mags: List[torch.Tensor] = []
        if self.sampler is not None:
            mags = [dist.rsample(batch_shape[:1], self.same_on_batch) for dist in self.sampler]
        if self.mapper is not None:
            mags = [mapping(mag) for mapping, mag in zip(self.mapper, mags)]
        return list(mags)

    # Change to named tuple
    def generate_parameters(self, input: torch.Tensor) -> Parameters:
        probs = self.get_batch_probabilities(input)
        mags = self.get_param_magnitudes(input)
        return Parameters(probs.bool(), mags)

    def forwad_transform(self, input: torch.Tensor, params: Parameters) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor, params: Optional[Parameters] = None) -> torch.Tensor:
        if params is None:
            params = self.generate_parameters(input)
        if (params.probs == 0).all():
            return input
        if (params.probs == 1).all():
            return self.forwad_transform(input, params)
        inp = input[params.probs]
        out = self.forwad_transform(inp, params)
        input[params.probs] = out
        return input


class IntensityAugmentOperation(AugmentOperation):
    """ """

    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = None,
        mapper: Optional[List[Callable]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
    ):
        super().__init__(
            p=p,
            p_batch=p_batch,
            sampler=sampler,
            mapper=mapper,
            gradients_estimator=gradients_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(self, input: torch.Tensor, params: Parameters) -> torch.Tensor:
        mag = [_mag[params.probs] for _mag in params.magnitudes]
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
        sampler: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = None,
        mapper: Optional[List[Callable]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            p=p,
            p_batch=p_batch,
            sampler=sampler,
            mapper=mapper,
            gradients_estimator=gradients_estimator,
            same_on_batch=same_on_batch,
        )
        self.return_transform = return_transform

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def inverse_transform(self, input: torch.Tensor, transform: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(  # type: ignore
        self, input: torch.Tensor, params: Parameters
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = [_mag[params.probs] for _mag in params.magnitudes]
        if self.gradients_estimator is not None:
            with torch.no_grad():
                trans_mat = self.compute_transform(input, mag)
                out = self.apply_transform(input, trans_mat)
            out = self.gradients_estimator.apply(
                input, out, lambda x, shape: self.inverse_transform(x, trans_mat, shape)
            )
        else:
            trans_mat = self.compute_transform(input, mag)
            out = self.apply_transform(input, trans_mat)
        return out, trans_mat

    def forward(  # type: ignore
        self, input: torch.Tensor, params: Optional[Parameters] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if params is None:
            params = self.generate_parameters(input)
        if (params.probs == 0).all():
            out, trans_mat = input, eye_like(3, input)
        elif (params.probs == 1).all():
            out, trans_mat = self.forwad_transform(input, params)
        else:
            inp = input[params.probs]
            in_trans_mat = eye_like(3, input)
            _out, _trans_mat = self.forwad_transform(inp, params)
            input[params.probs] = _out
            in_trans_mat[params.probs] = _trans_mat
            out, trans_mat = input, in_trans_mat
        if self.return_transform:
            return out, trans_mat
        return out


class PerspectiveAugmentOperation(GeometricAugmentOperation):
    """Base class for applying geometric data augmentation methods.

    Allowing apply and inverse transformations made by computed transform matrices.
    """
