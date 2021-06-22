from typing import Callable, List, NamedTuple, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Function

from kornia.augmentation.core.sampling import DynamicBernoulli, DynamicSampling, DynamicUniform
from kornia.geometry.epipolar.numeric import eye_like

Parameters = NamedTuple("Parameters", [('probs', torch.Tensor), ('magnitudes', List[torch.Tensor])])


class AugmentOperation(nn.Module):
    """ """

    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler_list: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = [],
        gradient_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
    ):
        super().__init__()
        self.gradient_estimator = gradient_estimator
        self.same_on_batch = same_on_batch
        self.prob_dist = DynamicBernoulli(p, freeze_dtype=True)
        self.prob_batch_dist = DynamicBernoulli(p_batch, freeze_dtype=True)
        self.sampler_list: Optional[nn.ModuleList]
        if sampler_list is not None:
            self.sampler_list = nn.ModuleList([self._make_sampler(dist) for dist in sampler_list])
        else:
            self.sampler_list = None

    def _make_sampler(self, sampler: Union[Tuple[float, float], DynamicSampling]) -> DynamicSampling:
        _sampler: DynamicSampling
        if isinstance(sampler, (list, tuple)):
            _sampler = DynamicUniform(*sampler)
        else:
            _sampler = sampler
        return _sampler

    def distribution_entropy(self, reduce: Optional[str] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.sampler_list is None:
            raise NotImplementedError(
                f"This method is invalid since `sampler_list` is passed as None.")
        dists = [dist.entropy() for dist in self.sampler_list]
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
        batch_probs = self.prob_batch_dist.rsample((1,), self.same_on_batch).squeeze()
        if batch_probs.bool().item():
            probs = self.prob_dist.rsample(batch_shape[:1], self.same_on_batch).squeeze()
            probs = probs.unsqueeze(0) if probs.dim() == 0 else probs
        else:
            probs = batch_probs.expand(batch_shape[0])
        return probs

    def get_param_magnitudes(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Parameter sampling methods."""
        if self.sampler_list is None:
            raise NotImplementedError(
                f"This method may need to be overrided since `sampler_list` is passed as None.")
        batch_shape = input.shape
        mags: List[torch.Tensor] = []
        mags = [dist.rsample(batch_shape[:1], self.same_on_batch) for dist in self.sampler_list]
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
        sampler_list: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = [],
        gradient_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
    ):
        super().__init__(
            p=p,
            p_batch=p_batch,
            sampler_list=sampler_list,
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
        )

    def apply_transform(self, input: torch.Tensor, magnitude: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(self, input: torch.Tensor, params: Parameters) -> torch.Tensor:
        mag = [_mag[params.probs] for _mag in params.magnitudes]
        out = input.clone()
        if self.gradient_estimator is not None:
            with torch.no_grad():
                out[params.probs] = self.apply_transform(input[params.probs], mag)
            out = self.gradient_estimator.apply(input, out)
        else:
            out[params.probs] = self.apply_transform(out[params.probs], mag)
        return out


class GeometricAugmentOperation(AugmentOperation):
    """Base class for applying geometric data augmentation methods.

    Allowing apply and inverse transformations made by computed transform matrices.
    """

    def __init__(
        self,
        p: torch.Tensor,
        p_batch: torch.Tensor,
        sampler_list: Optional[List[Union[Tuple[float, float], DynamicSampling]]] = [],
        gradient_estimator: Optional[Function] = None,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            p=p,
            p_batch=p_batch,
            sampler_list=sampler_list,
            gradient_estimator=gradient_estimator,
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
        if self.gradient_estimator is not None:
            with torch.no_grad():
                trans_mat = self.compute_transform(input, mag)
                out = self.apply_transform(input, trans_mat)
            out = self.gradient_estimator.apply(
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
    """Base class for applying shape-non-persistent geometric data augmentation methods.

    Allowing apply and inverse transformations made by computed transform matrices.
    """
