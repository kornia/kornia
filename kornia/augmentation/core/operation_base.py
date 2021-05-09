from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import Distribution

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
        magnitude_dist: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                                 List[SmartSampling]]] = None,
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False
    ):
        super().__init__()
        if magnitude_mapping is not None:
            self.magnitude_mapping = self._make_magnitude_mapping(magnitude_mapping)
        else:
            self.magnitude_mapping = None
        self.gradients_estimator = gradients_estimator
        self.same_on_batch = same_on_batch
        self.prob_dist = SmartBernoulli(p, freeze_dtype=True)
        if magnitude_dist is not None:
            self.magnitude_dist = self._make_magnitude_dist(magnitude_dist)
        else:
            self.magnitude_dist = None

    def _make_magnitude_mapping(self, magnitude_mapping: Union[Callable, List[Callable]]) -> List[Callable]:
        if callable(magnitude_mapping):
            return [magnitude_mapping]
        if isinstance(magnitude_mapping, (tuple, list)):
            return magnitude_mapping
        raise ValueError

    def _make_magnitude_dist_one(self, magnitude_dist: Union[Tuple[float, float], SmartSampling]) -> Distribution:
        if isinstance(magnitude_dist, (list, tuple)):
            _magnitude_dist = SmartUniform(
                torch.tensor(magnitude_dist[0]),
                torch.tensor(magnitude_dist[1])
            )
        else:
            _magnitude_dist = magnitude_dist
        return _magnitude_dist

    def _make_magnitude_dist(
            self, magnitude_dist: Union[Tuple[float, float], List[Tuple[float, float]], List[SmartSampling],
                                        SmartSampling]
    ) -> List[Distribution]:
        """Make a list of distributions according to the parameters."""
        if (
            isinstance(magnitude_dist, (list, tuple)) and len(magnitude_dist) == 2 and
            isinstance(magnitude_dist[0], (int, float)) and isinstance(magnitude_dist[0], (int, float))
        ) or isinstance(magnitude_dist, (SmartSampling,)):
            magnitude_dist = [magnitude_dist]
        return [self._make_magnitude_dist_one(dist) for dist in magnitude_dist]

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Optional[torch.Tensor]]:
        probs = self.prob_dist.rsample(batch_shape[:1], self.same_on_batch).squeeze()
        mags = None
        if self.magnitude_dist is not None:
            mags = [dist.rsample(batch_shape[:1], self.same_on_batch) for dist in self.magnitude_dist]
        if self.magnitude_mapping is not None:
            mags = [mapping(mag) for mapping, mag in zip(self.magnitude_mapping, mags)]
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
        out = self.forwad_transform(input, params)
        input[params['probs']] = out
        return input


class IntensityAugmentOperation(AugmentOperation):
    """
    """
    def __init__(
        self,
        p: torch.Tensor,
        magnitude_dist: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                                 List[SmartSampling]]] = None,
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False
    ):
        super().__init__(
            p=p, magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )

    def compute_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
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
    """
    """
    def __init__(
        self,
        p: torch.Tensor,
        magnitude_dist: Optional[Union[Tuple[float, float], List[Tuple[float, float]], SmartSampling,
                                 List[SmartSampling]]] = None,
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None,
        gradients_estimator: Optional[Function] = None,
        same_on_batch: bool = False
    ):
        super().__init__(
            p=p, magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )

    def compute_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forwad_transform(self, input: torch.Tensor, params: Dict[str, Optional[torch.Tensor]]):
        if params['magnitudes'] is None:
            mag = None
        else:
            mag = [_mag[params['probs']] for _mag in params['magnitudes']]
            mag = mag[0] if len(mag) == 1 else mag
        if self.gradients_estimator is not None:
            with torch.no_grad():
                trans_mat = self.compute_transform(input, mag)
                out = self.apply_transform(input, trans_mat)
            out = self.gradients_estimator.apply(input, out)
        else:
            trans_mat = self.compute_transform(input, mag)
            out = self.apply_transform(input, trans_mat)
        return out
