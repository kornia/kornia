from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.constants import BorderType, Resample
from kornia.filters import motion_blur


class RandomMotionBlur(IntensityAugmentationBase2D):
    r"""Perform motion blur on 2D images (4D tensor).

    .. image:: _static/img/RandomMotionBlur.png

    Args:
        p: probability of applying the transformation.
        kernel_size: motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle: angle of the motion blur in degrees (anti-clockwise rotation).
            If float, it will generate the value from (-angle, angle).
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.
        border_type: the padding mode to be applied before convolving.
            CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3.
        resample: the interpolation mode.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

        Please set ``resample`` to ``'bilinear'`` if more meaningful gradients wanted.

    .. note::
        This function internally uses :func:`kornia.filters.motion_blur`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.ones(1, 1, 5, 5)
        >>> motion_blur = RandomMotionBlur(3, 35., 0.5, p=1.)
        >>> motion_blur(input)
        tensor([[[[0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561],
                  [0.5773, 1.0000, 1.0000, 1.0000, 0.7561]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomMotionBlur(3, 35., 0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        angle: Union[Tensor, float, Tuple[float, float]],
        direction: Union[Tensor, float, Tuple[float, float]],
        border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
        resample: Union[str, int, Resample] = Resample.NEAREST.name,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.MotionBlurGenerator(kernel_size, angle, direction)
        self.flags = dict(border_type=BorderType.get(border_type), resample=Resample.get(resample))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        # sample a kernel size
        kernel_size_list: List[int] = params["ksize_factor"].tolist()
        idx: int = cast(int, torch.randint(len(kernel_size_list), (1,)).item())
        return motion_blur(
            input,
            kernel_size=kernel_size_list[idx],
            angle=params["angle_factor"],
            direction=params["direction_factor"],
            border_type=flags["border_type"].name.lower(),
            mode=flags["resample"].name.lower(),
        )
