from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.autograd import Function

from kornia.augmentation.core.sampling import DynamicSampling
from kornia.geometry.transform import affine
from kornia.constants import BorderType, Resample, SamplePadding
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_shear_matrix, _compute_tensor_center

from .base import GeometricAugmentOperation

__all__ = [
    "ShearAugment",
    "RotationAugment",
]


class ShearAugment(GeometricAugmentOperation):
    """Perform shear augmentation.

    Args:
        x_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling x-direction shearing
            parameters to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0., 1.).
        y_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling y-direction shearing
            parameters to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0., 1.).
        x_mapper(Union[Tuple[float, float], Callable]], Optional): the mapping function to map the sampled x
            parameters to any range. If a tuple (a, b), it will map to (a, b) by `torch.clamp` by default, in which
            ``a`` and ``b`` can be None to indicate infinity. Otherwise, it will by mapped by the provided function.
            Default is None.
        y_mapper(Union[Tuple[float, float], Callable]], Optional): the mapping function to map the sampled y
            parameters to any range. If a tuple (a, b), it will map to (a, b) by `torch.clamp` by default, in which
            ``a`` and ``b`` can be None to indicate infinity. Otherwise, it will by mapped by the provided function.
            Default is None.
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        resample (int, str or kornia.Resample): resample mode from "nearest" (0) or "bilinear" (1).
            Default: Resample.BILINEAR.
        padding_mode (int, str or kornia.SamplePadding): padding mode from "zeros" (0), "border" (1)
            or "refection" (2). Default: SamplePadding.ZEROS.
        align_corners(bool): interpolation flag. Default: False.
        p (float): probability of the image being flipped. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.

    Examples:
        >>> a = ShearAugment(p=1.)
        >>> out = a(torch.randn(2, 3, 100, 100))
        >>> out.shape
        torch.Size([2, 3, 100, 100])

        >>> a = ShearAugment((0., 1.), (0.1, .9), same_on_batch=True, p=1.)
        >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
        >>> (out[0] == out[1]).all()
        tensor(True)

        Custom mapping with 'torch.tanh' and DynamicGaussian:
        >>> from kornia.augmentation.core.sampling import DynamicGaussian
        >>> a = ShearAugment(
        ...     x_sampler=DynamicGaussian(torch.tensor(1.), torch.tensor(1.)),
        ...     y_sampler=(0., 1.),
        ...     x_mapper=lambda x: torch.tanh(x) * 100,
        ...     y_mapper=lambda x: torch.tanh(x) * 100,
        ...     same_on_batch=True, p=1.)
        >>> out = a(torch.randn(1, 3, 100, 100).repeat(2, 1, 1, 1))
        >>> (out[0] == out[1]).all()
        tensor(True)
    """

    def __init__(
        self,
        x_sampler: Union[Tuple[float, float], DynamicSampling] = (0.0, 1.0),
        y_sampler: Union[Tuple[float, float], DynamicSampling] = (0.0, 1.0),
        x_mapper: Optional[Union[Tuple[float, float], Callable]] = None,
        y_mapper: Optional[Union[Tuple[float, float], Callable]] = None,
        gradient_estimator: Optional[Function] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        align_corners: bool = False,
        p: float = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[x_sampler, y_sampler],
            mapper_list=[x_mapper, y_mapper],
            same_on_batch=same_on_batch,
            gradient_estimator=gradient_estimator,
            return_transform=return_transform,
        )
        self.resample = Resample.get(resample).name.lower()
        self.padding_mode = SamplePadding.get(padding_mode).name.lower()
        self.align_corners = align_corners

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:
        mag = torch.stack([magnitudes[0], magnitudes[1]], dim=1)
        return _compute_shear_matrix(mag)

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return affine(
            input,
            transform[..., :2, :3],
            mode=self.resample,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class RotationAugment(GeometricAugmentOperation):
    """Perform rotation augmentation.

    Args:
        rot_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling rotation
            degrees to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0., 1.).
        rot_mapper(Union[Tuple[float, float], Callable]], Optional): the mapping function to map the sampled rotation
            degrees to any range. If a tuple (a, b), it will map to (a, b) by `torch.clamp` by default, in which
            ``a`` and ``b`` can be None to indicate infinity. Otherwise, it will by mapped by the provided function.
            Default is None.
        gradient_estimator(Function, optional): gradient estimator for this operation. Default is None.
        resample (int, str or kornia.Resample): resample mode from "nearest" (0) or "bilinear" (1).
            Default: Resample.BILINEAR.
        padding_mode (int, str or kornia.SamplePadding): padding mode from "zeros" (0), "border" (1)
            or "refection" (2). Default: SamplePadding.ZEROS.
        align_corners(bool): interpolation flag. Default: False.
        p (float): probability of the image being flipped. Default value is 0.5.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation wont be concatenated.

    Examples:
        >>> a = RotationAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> out.mean().backward()

        Sampling with Gaussian:
        >>> from kornia.augmentation.core.sampling import DynamicGaussian
        >>> a = RotationAugment(DynamicGaussian(torch.tensor(1.), torch.tensor(1.)), p=1.)
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
        >>> a = RotationAugment(p=1., gradient_estimator=STEFunction)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.mean().backward()
    """

    def __init__(
        self,
        rot_sampler: Union[Tuple[float, float], DynamicSampling] = (-180., 180.),
        rot_mapper: Optional[Union[Tuple[float, float], Callable]] = None,
        gradient_estimator: Optional[Function] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        align_corners: bool = False,
        p: float = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[rot_sampler],
            mapper_list=[rot_mapper],
            same_on_batch=same_on_batch,
            gradient_estimator=gradient_estimator,
            return_transform=return_transform,
        )
        self.resample = Resample.get(resample).name.lower()
        self.padding_mode = SamplePadding.get(padding_mode).name.lower()
        self.align_corners = align_corners

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:
        center: torch.Tensor = _compute_tensor_center(input)
        rotation_mat: torch.Tensor = _compute_rotation_matrix(magnitudes[0], center.expand(magnitudes[0].shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: torch.Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]
        return trans_mat

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        return affine(
            input,
            transform[..., :2, :3],
            mode=self.resample,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def inverse_transform(self, input: torch.Tensor, transform: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        return affine(
            input,
            transform.inverse()[..., :2, :3],
            mode=self.resample,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
