from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.intensity.base import IntensityAugmentationBase3D
from kornia.constants import BorderType, Resample
from kornia.core import Tensor
from kornia.filters import motion_blur3d


class RandomMotionBlur3D(IntensityAugmentationBase3D):
    r"""Apply random motion blur on 3D volumes (5D tensor).

    Args:
        p: probability of applying the transformation.
        kernel_size: motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle: Range of degrees to select from.
            If angle is a number, then yaw, pitch, roll will be generated from the range of (-angle, +angle).
            If angle is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If angle is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If angle is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        direction: forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.
        border_type: the padding mode to be applied before convolving.
            CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, D, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 5, 5)
        >>> motion_blur = RandomMotionBlur3D(3, 35., 0.5, p=1.)
        >>> motion_blur(input)
        tensor([[[[[0.1654, 0.4772, 0.2004, 0.3566, 0.2613],
                   [0.4557, 0.3131, 0.4809, 0.2574, 0.2696],
                   [0.2721, 0.5998, 0.3956, 0.5363, 0.1541],
                   [0.3006, 0.4773, 0.6395, 0.2856, 0.3989],
                   [0.4491, 0.5595, 0.1836, 0.3811, 0.1398]],
        <BLANKLINE>
                  [[0.1843, 0.4240, 0.3370, 0.1231, 0.2186],
                   [0.4047, 0.3332, 0.1901, 0.5329, 0.3023],
                   [0.3070, 0.3088, 0.4807, 0.4928, 0.2590],
                   [0.2416, 0.4614, 0.7091, 0.5237, 0.1433],
                   [0.1582, 0.4577, 0.2749, 0.1369, 0.1607]],
        <BLANKLINE>
                  [[0.2733, 0.4040, 0.4396, 0.2284, 0.3319],
                   [0.3856, 0.6730, 0.4624, 0.3878, 0.3076],
                   [0.4307, 0.4217, 0.2977, 0.5086, 0.5406],
                   [0.3686, 0.2778, 0.5228, 0.7592, 0.6455],
                   [0.2033, 0.3014, 0.4898, 0.6164, 0.3117]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomMotionBlur3D(3, 35., 0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        angle: Union[
            Tensor,
            float,
            Tuple[float, float, float],
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ],
        direction: Union[Tensor, float, Tuple[float, float]],
        border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
        resample: Union[str, int, Resample] = Resample.NEAREST.name,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.flags = {"border_type": BorderType.get(border_type), "resample": Resample.get(resample)}
        self._param_generator = rg.MotionBlurGenerator3D(kernel_size, angle, direction)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        kernel_size = int(params["ksize_factor"].unique().item())
        angle = params["angle_factor"]
        direction = params["direction_factor"]
        return motion_blur3d(
            input,
            kernel_size,
            angle,
            direction,
            self.flags["border_type"].name.lower(),
            self.flags["resample"].name.lower(),
        )
