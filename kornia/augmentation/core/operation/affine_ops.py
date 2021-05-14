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
    SmartUniform,
)
from kornia.augmentation.core.gradient_estimator import (
    STEFunction,
    StraightThroughEstimator
)
from .base import GeometricAugmentOperation


class ShearX(GeometricAugmentOperation):
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
    >>> a = ShearX(mapper=lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)

    Custom mapping:
    >>> from kornia.augmentation.core.smart_sampling import SmartGaussian
    >>> a = ShearX(
    ... mapper=lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.,
    ... sampler=SmartGaussian(torch.tensor(1.), torch.tensor(1.)))
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)
    """
    def __init__(
        self,
        sampler: Union[Tuple[float, float], SmartSampling] = (0., 1.),
        mapper: Optional[Union[Callable, List[Callable]]] = None, mode: str = 'bilinear',
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

    def compute_transform(self, input: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        magnitudes = torch.stack([magnitudes, torch.zeros_like(magnitudes)], dim=1)
        return _compute_shear_matrix(magnitudes)

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return affine(input, transform[..., :2, :3], mode=self.mode, padding_mode=self.padding_mode,
                      align_corners=self.align_corners)


class Rotation(GeometricAugmentOperation):
    """
    >>> a = Rotation(p=1.)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])
    >>> out.mean().backward()

    Gradients Estimation - 1:
    >>> a = Rotation(p=1.)
    >>> input = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
    >>> with torch.no_grad():
    ...     out = a(input)
    >>> out_est = StraightThroughEstimator()(input, out)
    >>> out_est.mean().backward()

    Gradients Estimation - 2:
    >>> a = Rotation(p=1., gradients_estimator=STEFunction)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.mean().backward()
    """
    def __init__(
        self,
        sampler: Union[Tuple[float, float], SmartSampling] = (0., 360.),
        mapper: Optional[Union[Callable, List[Callable]]] = None, mode: str = 'bilinear',
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

    def compute_transform(self, input: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:

        center: torch.Tensor = _compute_tensor_center(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix(
            magnitudes, center.expand(magnitudes.shape[0], -1))

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
