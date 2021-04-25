from typing import Optional, Callable, Dict, Union, Tuple, cast

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import (
    Distribution,
    RelaxedBernoulli,
    Uniform,
)

from .sampling import (
    SmartSampling,
    SmartBernoulli,
    SmartUniform,
    SmartGaussian
)
from .gradient_estimator import StraightThroughEstimator

from kornia.geometry.transform import (
    shear,
    get_perspective_transform,
    warp_perspective,
)

from kornia.enhance import (
    equalize
)


class AugmentOperation(nn.Module):
    """
    """
    def __init__(
        self,
        p: torch.Tensor,
        magnitude_dist: Optional[SmartSampling] = None,
        magnitude_mapping: Optional[Union[Tuple[float, float], Callable]] = None,
        gradients_estimation: Optional[Function] = None,
        # TODO: remove ''is_train_dist'' parameter, to make the conversion from torch.Tensor
        # to nn.Param automatically
        is_train_dist: bool = True,
        same_on_batch: bool = False
    ):
        super().__init__()
        self.magnitude_mapping = magnitude_mapping
        self.gradients_estimation = gradients_estimation
        self.same_on_batch = same_on_batch

        self.prob_dist = SmartBernoulli(p)
        self.magnitude_dist = self._make_magnitude_dist(magnitude_dist)
        if self.magnitude_dist is not None:
            for param in self.magnitude_dist.parameters():
                param.requires_grad = is_train_dist

    def _make_magnitude_dist(self, magnitude_dist: Distribution) -> Distribution:
        if self.magnitude_mapping is not None:
            assert isinstance(self.magnitude_mapping, (list, tuple)) or callable(self.magnitude_mapping)
        if magnitude_dist is None and isinstance(self.magnitude_mapping, (list, tuple)):
            _magnitude_dist = SmartUniform(
                torch.tensor(self.magnitude_mapping[0]),
                torch.tensor(self.magnitude_mapping[1])
            )
        else:
            _magnitude_dist = magnitude_dist
        return _magnitude_dist

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, Optional[torch.Tensor]]:
        probs = self.prob_dist.rsample(batch_shape[:1], self.same_on_batch).squeeze()

        mags = None
        if self.magnitude_dist is not None:
            mags = self.magnitude_dist.rsample(batch_shape[:1], self.same_on_batch)
        if self.magnitude_mapping is not None and not isinstance(self.magnitude_mapping, (list, tuple)):
            mags = self.magnitude_mapping(mags)
        return {"probs": probs.bool(), "magnitudes": mags}

    def compute_transformation(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self.generate_parameters(input.shape)
        if (params['probs'] == 0).all():
            return input
        if params['magnitudes'] is None:
            mag = None
        else:
            mag = params['magnitudes'][params['probs']]
        inp = input[params['probs']]
        if self.gradients_estimation is not None:
            with torch.no_grad():
                out = self.apply_transform(inp, mag)
            out = self.gradients_estimation.apply(inp, out)
        else:
            out = self.apply_transform(inp, mag)
        input[params['probs']] = out
        return input


class ShearX(AugmentOperation):
    """
    >>> a = ShearX((0., 1.), p=1.)
    >>> out = a(torch.randn(2, 3, 100, 100))
    >>> out.shape
    torch.Size([2, 3, 100, 100])

    >>> a = ShearX((0., 1.), same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)

    Custom mapping with 'torch.tanh':
    >>> a = ShearX(lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)

    Custom mapping:
    >>> a = ShearX(
    ... lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.,
    ... magnitude_dist=SmartGaussian(torch.tensor(1.), torch.tensor(1.)))
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)
    """
    def __init__(
        self, magnitude_mapping: Union[Tuple[float, float], Callable] = (0., 1.), mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = False, p: float = 0.5, same_on_batch: bool = False,
        magnitude_dist: Optional[SmartSampling] = None, is_train_dist: bool = False,
        gradients_estimation: Optional[Function] = None
    ):
        if magnitude_dist is None:
            magnitude_dist = SmartUniform(torch.tensor(0.), torch.tensor(1.))
        super().__init__(
            torch.tensor(p), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            is_train_dist=is_train_dist, same_on_batch=same_on_batch, gradients_estimation=gradients_estimation
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def apply_transform(self, input: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        magnitudes = torch.stack([magnitudes, torch.zeros_like(magnitudes)], dim=1)
        return shear(input, magnitudes, mode=self.mode, padding_mode=self.padding_mode,
                     align_corners=self.align_corners)


class Perspective(AugmentOperation):
    """
    >>> a = Perspective(p=1.)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])
    >>> loss = out.mean()
    >>> loss.backward()
    """
    def __init__(
        self, magnitude_mapping: Union[Tuple[float, float], Callable] = (0.3, 0.7), p: float = 0.5,
        same_on_batch: bool = False, mode: str = 'bilinear', align_corners: bool = True,
        magnitude_dist: Optional[SmartSampling] = None, is_train_dist: bool = False,
        gradients_estimation: Optional[Function] = StraightThroughEstimator
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            gradients_estimation=gradients_estimation,
            is_train_dist=is_train_dist, same_on_batch=same_on_batch
        )
        self.mode = mode
        self.align_corners = align_corners
        self.rand_val = SmartUniform(torch.tensor(0.), torch.tensor(1.))

    def compute_transform(self, input: torch.Tensor, magnitudes: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = input.shape

        start_points: torch.Tensor = torch.tensor([[
            [0., 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]], device=input.device, dtype=input.dtype).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = magnitudes * width / 2
        fy = magnitudes * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

        with torch.no_grad():
            # No need to be trainable here. We wish to keep it as a uniform distribution
            rand_val = self.rand_val.rsample(input.shape[:1], self.same_on_batch).to(input)

        pts_norm = torch.tensor([[
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]], device=input.device, dtype=input.dtype)
        end_points = start_points + factor * rand_val * pts_norm

        transform: torch.Tensor = get_perspective_transform(start_points, end_points)
        return transform

    def apply_transform(self, input: torch.Tensor, magnitudes: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        out_data = warp_perspective(
            input, transform, (height, width),
            mode=self.mode, align_corners=self.align_corners)
        return out_data


class Equalize(AugmentOperation):
    """
    >>> a = Equalize(1.)
    >>> out = a(torch.ones(2, 3, 100, 100) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])

    # Backprop with gradients estimator
    >>> a = Equalize(1., gradients_estimation=StraightThroughEstimator)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> loss = (out - torch.ones(2, 3, 100, 100)).mean()
    >>> loss.backward()
    """
    def __init__(
        self, p: float = 0.5, same_on_batch: bool = False, is_train_dist: bool = False,
        gradients_estimation: Optional[Function] = StraightThroughEstimator
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=None, magnitude_mapping=None, gradients_estimation=gradients_estimation,
            is_train_dist=is_train_dist, same_on_batch=same_on_batch
        )

    def apply_transform(self, input: torch.Tensor, _: Optional[torch.Tensor]) -> torch.Tensor:
        return equalize(input)
