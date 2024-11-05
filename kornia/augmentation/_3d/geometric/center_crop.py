from typing import Any, Dict, Optional, Tuple, Union, cast

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.constants import Resample
from kornia.core import Tensor
from kornia.geometry import crop_by_transform_mat3d, get_perspective_transform3d


class CenterCrop3D(GeometricAugmentationBase3D):
    r"""Apply center crop on 3D volumes (5D tensor).

    Args:
        p: probability of applying the transformation for the whole batch.
        size (Tuple[int, int, int] or int): Desired output size (out_d, out_h, out_w) of the crop.
            If integer, out_d = out_h = out_w = size.
            If Tuple[int, int, int], out_d = size[0], out_h = size[1], out_w = size[2].
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False).

    Shape:
        - Input: :math:`(C, D, H, W)` or :math:`(B, C, D, H, W)`, Optional: :math:`(B, 4, 4)`
        - Output: :math:`(B, C, out_d, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 4, 4)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 2, 4, 6)
        >>> inputs
        tensor([[[[[-1.1258, -1.1524, -0.2506, -0.4339,  0.8487,  0.6920],
                   [-0.3160, -2.1152,  0.3223, -1.2633,  0.3500,  0.3081],
                   [ 0.1198,  1.2377,  1.1168, -0.2473, -1.3527, -1.6959],
                   [ 0.5667,  0.7935,  0.5988, -1.5551, -0.3414,  1.8530]],
        <BLANKLINE>
                  [[ 0.7502, -0.5855, -0.1734,  0.1835,  1.3894,  1.5863],
                   [ 0.9463, -0.8437, -0.6136,  0.0316, -0.4927,  0.2484],
                   [ 0.4397,  0.1124,  0.6408,  0.4412, -0.1023,  0.7924],
                   [-0.2897,  0.0525,  0.5229,  2.3022, -1.4689, -1.5867]]]]])
        >>> aug = CenterCrop3D(2, p=1.)
        >>> aug(inputs)
        tensor([[[[[ 0.3223, -1.2633],
                   [ 1.1168, -0.2473]],
        <BLANKLINE>
                  [[-0.6136,  0.0316],
                   [ 0.6408,  0.4412]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = CenterCrop3D(24, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int, int]],
        align_corners: bool = True,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        # same_on_batch is always True for CenterCrop
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=True, p_batch=p, keepdim=keepdim)
        if isinstance(size, tuple):
            self.size = (size[0], size[1], size[2])
        elif isinstance(size, int):
            self.size = (size, size, size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int int). Got: {size}.")
        self.flags = {"align_corners": align_corners, "resample": Resample.get(resample)}

    def generate_parameters(self, batch_shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        return rg.center_crop_generator3d(
            batch_shape[0], batch_shape[-3], batch_shape[-2], batch_shape[-1], self.size, device=self.device
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        transform: Tensor = get_perspective_transform3d(params["src"].to(input), params["dst"].to(input))
        transform = transform.expand(input.shape[0], -1, -1)
        return transform

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        transform = cast(Tensor, transform)
        return crop_by_transform_mat3d(
            input, transform, self.size, mode=flags["resample"].name.lower(), align_corners=flags["align_corners"]
        )
