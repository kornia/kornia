from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function

from kornia.geometry.transform import (
    affine,
)
from kornia.geometry.transform.affwarp import (
    _compute_shear_matrix,
    _compute_tensor_center,
    _compute_rotation_matrix,
)
from kornia.augmentation.core.smart_sampling import (
    SmartSampling,
)
from .base import GeometricAugmentOperation


class ShearAugment(GeometricAugmentOperation):
    """
    >>> a = ShearAugment(p=1.)
    >>> out = a(torch.randn(2, 3, 100, 100))
    >>> out.shape
    torch.Size([2, 3, 100, 100])

    >>> a = ShearAugment([(0., 1.), (0.1, .9)], same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)

    Custom mapping with 'torch.tanh' and SmartGaussian:
    >>> from kornia.augmentation.core.smart_sampling import SmartGaussian
    >>> a = ShearAugment(
    ...     sampler=[SmartGaussian(torch.tensor(1.), torch.tensor(1.)), (0., 1.)],
    ...     mapper=[lambda x: torch.tanh(x) * 100, lambda x: torch.tanh(x) * 100],
    ...     same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)
    """
    def __init__(
        self,
        sampler: List[Union[Tuple[float, float], SmartSampling]] = [(0., 1.), (0., 1.)],
        mapper: Optional[List[Callable]] = None, mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = False, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), torch.tensor(1.), sampler=sampler, mapper=mapper,
            same_on_batch=same_on_batch, gradients_estimator=gradients_estimator
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:
        magnitudes = torch.stack([magnitudes[0], magnitudes[1]], dim=1)
        return _compute_shear_matrix(magnitudes)

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return affine(input, transform[..., :2, :3], mode=self.mode, padding_mode=self.padding_mode,
                      align_corners=self.align_corners)


class RotationAugment(GeometricAugmentOperation):
    """
    >>> a = RotationAugment(p=1.)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])
    >>> out.mean().backward()

    Sampling with Gaussian:
    >>> from kornia.augmentation.core.smart_sampling import SmartGaussian
    >>> a = RotationAugment(SmartGaussian(torch.tensor(1.), torch.tensor(1.)), p=1.)
    >>> out = a(torch.ones(20, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([20, 3, 100, 100])
    >>> out.mean().backward()

    Gradients Estimation - 1:
    >>> from kornia.augmentation.core.gradient_estimator import StraightThroughEstimator
    >>> a = RotationAugment(p=1.)
    >>> input = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
    >>> with torch.no_grad():
    ...     out = a(input)
    >>> out_est = StraightThroughEstimator()(input, out)
    >>> out_est.mean().backward()

    Gradients Estimation - 2:
    >>> from kornia.augmentation.core.gradient_estimator import STEFunction
    >>> a = RotationAugment(p=1., gradients_estimator=STEFunction)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.mean().backward()
    """
    def __init__(
        self,
        sampler: Union[Tuple[float, float], SmartSampling] = (0., 360.),
        mapper: Optional[List[Callable]] = None, mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = False, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), torch.tensor(1.), sampler=[sampler], mapper=mapper,
            same_on_batch=same_on_batch, gradients_estimator=gradients_estimator
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:

        center: torch.Tensor = _compute_tensor_center(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix(
            magnitudes[0], center.expand(magnitudes[0].shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]
        return trans_mat

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return affine(input, transform[..., :2, :3], mode=self.mode, padding_mode=self.padding_mode,
                      align_corners=self.align_corners)

    def inverse_transform(self, input: torch.Tensor, transform: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        return affine(input, transform.inverse()[..., :2, :3], mode=self.mode, padding_mode=self.padding_mode,
                      align_corners=self.align_corners)
