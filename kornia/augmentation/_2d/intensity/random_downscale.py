from typing import Tuple, Union, Optional, Dict, Any

import torch
from torch import Tensor
from torch.nn.functional import interpolate

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE


class RandomDownScale(IntensityAugmentationBase2D):
    r"""Randomly downscale an image or a batch of images.

    Args:
        size (Union[int, Tuple[int, int]]): output spatial size of the downscaled image.
            If `int`, the output image will have the same size on both sides.
            If `Tuple[int, int]`, specifies the output size as `(height, width)`.
        random_var (float): percantage possible possible deviation from actuiual size value
        p (float): probability of applying the transformation.
        p_batch: probability for applying an augmentation to a batch. This param controls the augmentation
            probabilities batch-wise.
        same_on_batch: apply the same transformation across the batch.

    Returns:
        A downsampled tensor with the specified output size.

    Shape:
        - Input: `(B, C, H, W)`
        - Output: `(B, C, H', W')`

    Example:
        >>> input = torch.rand(1, 3, 256, 256)
        >>> downscale = RandomDownScale(size=128, p=0.5)
        >>> output = downscale(input)
        >>> assert output.size() == (1, 3, 128, 128)
    """
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        random_var: float = 0.5,
        same_on_batch: bool = False,
        p: float = 0.5,
        p_batch: float = 1.0, 
        keepdim: bool = False,
    ) -> None:
        super().__init__(same_on_batch=same_on_batch, p=p, p_batch=p_batch, keepdim=keepdim)

        KORNIA_CHECK(0. <= random_var < 1., "Random variance should be in range [0;1)")
        self.random_var = random_var
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None
    ) -> Tensor:
        random_scale_factor = 1 + torch.rand(1).item() * self.random_var
        scaled_size = (int(self.size[0] * random_scale_factor), int(self.size[1] * random_scale_factor))

        return interpolate(
            input,
            size=scaled_size,
            mode='bilinear',
            align_corners=False
        )
