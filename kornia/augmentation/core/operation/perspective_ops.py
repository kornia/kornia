from typing import List, Optional, Tuple, Union

import torch
from torch.autograd import Function

from kornia.augmentation.core.sampling import DynamicSampling, DynamicUniform
from kornia.geometry.transform import (
    bbox_generator,
    crop_by_transform_mat,
    get_perspective_transform,
    warp_perspective
)
from kornia.constants import Resample

from .base import PerspectiveAugmentOperation, Parameters

__all__ = [
    "PerspectiveAugment",
    "CropAugment",
]


class PerspectiveAugment(PerspectiveAugmentOperation):
    """Perform perspective augmentation.

    Args:
        sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling perspective
            factors to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0.3, 0.7).
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
        >>> a = PerspectiveAugment(p=1.)
        >>> out = a(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 100, 100])
        >>> loss = out.mean()
        >>> loss.backward()
    """

    def __init__(
        self,
        sampler: Union[Tuple[float, float], DynamicSampling] = (0.3, 0.7),
        gradient_estimator: Optional[Function] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        align_corners: bool = True,
        p: float = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            torch.tensor(p),
            torch.tensor(1.0),
            sampler_list=[sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
            return_transform=return_transform,
        )
        self.resample = Resample.get(resample).name.lower()
        self.align_corners = align_corners
        self.rand_val = DynamicUniform(torch.tensor(0.0), torch.tensor(1.0))

    def generate_parameters(self, input: torch.Tensor) -> Parameters:
        p = super().generate_parameters(input)

        with torch.no_grad():
            # No need to be trainable here. We wish to keep it as a uniform distribution
            rand_val = self.rand_val.rsample(input.shape[:1], self.same_on_batch)
        p.magnitudes.append(rand_val)

        return p

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = input.shape

        start_points: torch.Tensor = torch.tensor(
            [[[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
            device=input.device,
            dtype=input.dtype,
        ).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        fx = magnitudes[0] * width / 2
        fy = magnitudes[0] * height / 2

        factor = torch.stack([fx, fy], dim=0).view(-1, 1, 2)

        pts_norm = torch.tensor([[[1, 1], [-1, 1], [-1, -1], [1, -1]]], device=input.device, dtype=input.dtype)
        end_points = start_points + factor * magnitudes[-1] * pts_norm  # the last magnitude is the rand_val

        transform: torch.Tensor = get_perspective_transform(start_points, end_points)
        return transform

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        _, _, height, width = input.shape
        out_data = warp_perspective(
            input, transform, (height, width), mode=self.resample,
            padding_mode='zeros', align_corners=self.align_corners)
        return out_data


class CropAugment(PerspectiveAugmentOperation):
    """Perform crop augmentation.

    Args:
        x_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling starting x-axis
            points to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0., 1.).
        y_sampler (List[Union[Tuple[float, float], DynamicSampling]]): sampler for sampling starting y-axis
            points to perform the transformation. If a tuple (a, b), it will sample from (a, b) uniformly.
            Otherwise, it will sample from the pointed sampling distribution. Default is (0., 1.).
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
        >>> crop = CropAugment((50, 50), p=1.)
        >>> out = crop(torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5)
        >>> out.shape
        torch.Size([2, 3, 50, 50])
        >>> loss = (out - 1).mean()
        >>> loss.backward()

        Gradients Estimation - 1:
        >>> from kornia.augmentation.core.gradient_estimator import STEFunction
        >>> crop = CropAugment((50, 50), p=1., gradient_estimator=STEFunction)
        >>> inp = torch.ones(2, 3, 100, 100, requires_grad=True) * 0.5
        >>> out = crop(inp)
        >>> out.mean().backward()
        >>> inp.grad
    """

    def __init__(
        self,
        size: Tuple[int, int],
        x_sampler: Union[Tuple[float, float], DynamicSampling] = (0.0, 1.0),
        y_sampler: Union[Tuple[float, float], DynamicSampling] = (0.0, 1.0),
        gradient_estimator: Optional[Function] = None,
        p: float = 0.5,
        same_on_batch: bool = False,
        return_transform: bool = False,
    ):
        super().__init__(
            torch.tensor(1.0),
            torch.tensor(p),
            sampler_list=[x_sampler, y_sampler],
            gradient_estimator=gradient_estimator,
            same_on_batch=same_on_batch,
            return_transform=return_transform,
        )
        self.size = size
        _crop_dst: torch.Tensor = torch.tensor(
            [[[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1], [0, size[0] - 1]]]
        )
        self.register_buffer("crop_dst", _crop_dst)

    def compute_transform(self, input: torch.Tensor, magnitudes: List[torch.Tensor]) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        x_diff = height - self.size[1] + 1
        y_diff = width - self.size[0] + 1
        x_start = torch.floor(magnitudes[0] * x_diff)
        y_start = torch.floor(magnitudes[1] * y_diff)
        w = x_start * 0 + self.size[1]
        h = y_start * 0 + self.size[0]
        crop_src = bbox_generator(x_start, y_start, w, h)
        return get_perspective_transform(crop_src, self.crop_dst.expand(batch_size, -1, -1))

    def apply_transform(self, input: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        out = crop_by_transform_mat(
            input, transform, self.size, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        return out

    def inverse_transform(self, input: torch.Tensor, transform: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        out = crop_by_transform_mat(
            input,
            transform.pinverse(),
            (output_shape[-2], output_shape[-1]),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return out
