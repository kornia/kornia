from typing import Optional, Callable, Dict, Union, Tuple, List, cast

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.distributions import Distribution

from kornia.geometry.transform import (
    warp_perspective,
    bbox_generator,
    get_perspective_transform,
    crop_by_transform_mat,
    affine,
)
from kornia.geometry.transform.affwarp import (
    _compute_shear_matrix,
    _compute_tensor_center,
    _compute_rotation_matrix,
)
from kornia.enhance import (
    equalize
)
from .smart_sampling import (
    SmartSampling,
    SmartUniform,
)
from .gradient_estimator import (
    STEFunction,
    StraightThroughEstimator
)
from .operation_base import GeometricAugmentOperation, IntensityAugmentOperation


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
    >>> a = ShearX(magnitude_mapping=lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.)
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)

    Custom mapping:
    >>> from kornia.augmentation.core.smart_sampling import SmartGaussian
    >>> a = ShearX(
    ... magnitude_mapping=lambda x: torch.tanh(x) * 100, same_on_batch=True, p=1.,
    ... magnitude_dist=SmartGaussian(torch.tensor(1.), torch.tensor(1.)))
    >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
    >>> (out[0] == out[1]).all()
    tensor(True)
    """
    def __init__(
        self,
        magnitude_dist: Union[Tuple[float, float], SmartSampling] = (0., 1.),
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None, mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = False, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
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
        magnitude_dist: Union[Tuple[float, float], SmartSampling] = (0., 360.),
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None, mode: str = 'bilinear',
        padding_mode: str = 'zeros', align_corners: bool = False, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
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


class Perspective(GeometricAugmentOperation):
    """
    >>> a = Perspective(p=1.)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])
    >>> loss = out.mean()
    >>> loss.backward()
    """
    def __init__(
        self,
        magnitude_dist: Union[Tuple[float, float], SmartSampling] = (0.3, 0.7),
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None, p: float = 0.5,
        same_on_batch: bool = False, mode: str = 'bilinear', align_corners: bool = True,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
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
            rand_val = self.rand_val.rsample(input.shape[:1], self.same_on_batch)

        pts_norm = torch.tensor([[
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ]], device=input.device, dtype=input.dtype)
        end_points = start_points + factor * rand_val * pts_norm

        transform: torch.Tensor = get_perspective_transform(start_points, end_points)
        return transform

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        out_data = warp_perspective(
            input, transform, (height, width),
            mode=self.mode, align_corners=self.align_corners)
        return out_data


class Crop(GeometricAugmentOperation):
    """
    >>> crop = Crop((50, 50), p=1.)
    >>> out = crop(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 50, 50])
    >>> loss = (out - 1).mean()
    >>> loss.backward()

    Gradients Estimation - 1:
    >>> crop = Crop((50, 50), p=1.)
    >>> input = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
    >>> with torch.no_grad():
    ...     out = crop(input)
    >>> out_est = StraightThroughEstimator()(input, out)
    >>> out_est.mean().backward()

    Gradients Estimation - 2:
    >>> crop = Crop((50, 50), p=1., gradients_estimator=STEFunction)
    >>> out = crop(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> out.mean().backward()
    """
    def __init__(
        self, size: Tuple[int, int], p: float = 0.5,
        magnitude_dist: Union[List[Tuple[float, float]], List[SmartSampling]] = [(0., 1.), (0., 1.)],
        magnitude_mapping: Optional[Union[Callable, List[Callable]]] = None, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = None
    ):
        super().__init__(
            torch.tensor(1.), magnitude_dist=magnitude_dist, magnitude_mapping=magnitude_mapping,
            gradients_estimator=gradients_estimator, same_on_batch=same_on_batch
        )
        self.size = size
        _crop_dst = torch.tensor([[
            [0, 0],
            [size[1] - 1, 0],
            [size[1] - 1, size[0] - 1],
            [0, size[0] - 1],
        ]])
        self.register_buffer("crop_dst", _crop_dst)

    def compute_transform(self, input: torch.Tensor, magnitudes: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        x_diff = height - self.size[1] + 1
        y_diff = width - self.size[0] + 1
        x_start = torch.floor(magnitudes[0] * x_diff)
        y_start = torch.floor(magnitudes[1] * y_diff)
        width = x_start * 0 + self.size[1]
        height = y_start * 0 + self.size[0]
        crop_src = bbox_generator(x_start, y_start, width, height)
        return get_perspective_transform(crop_src, self.crop_dst.expand(batch_size, -1, -1))

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        out = crop_by_transform_mat(
            input, transform, self.size, mode='bilinear',
            padding_mode='zeros', align_corners=True)
        return out


class Equalize(IntensityAugmentOperation):
    """
    >>> a = Equalize(1.)
    >>> out = a(torch.ones(2, 3, 100, 100) * 0.5)
    >>> out.shape
    torch.Size([2, 3, 100, 100])

    # Backprop with gradients estimator
    >>> a = Equalize(1., gradients_estimator=STEFunction)
    >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
    >>> loss = (out - torch.ones(2, 3, 100, 100)).mean()
    >>> loss.backward()
    """
    def __init__(
        self, p: float = 0.5, same_on_batch: bool = False,
        gradients_estimator: Optional[Function] = STEFunction
    ):
        super().__init__(
            torch.tensor(p), magnitude_dist=None, magnitude_mapping=None, gradients_estimator=gradients_estimator,
            same_on_batch=same_on_batch
        )

    def apply_transform(self, input: torch.Tensor, _: Optional[torch.Tensor]) -> torch.Tensor:
        return equalize(input)
