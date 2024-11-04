from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from kornia.constants import Resample
from kornia.core import Tensor
from kornia.geometry import deg2rad, get_affine_matrix3d, warp_affine3d


class RandomAffine3D(GeometricAugmentationBase3D):
    r"""Apply affine transformation 3D volumes (5D tensor).

    The transformation is computed so that the center is kept invariant.

    Args:
        degrees: Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal, vertical and
            depthical translations (dx,dy,dz). For example translate=(a, b, c), then
            horizontal shift will be randomly sampled in the range -img_width * a < dx < img_width * a
            vertical shift will be randomly sampled in the range -img_height * b < dy < img_height * b.
            depthical shift will be randomly sampled in the range -img_depth * c < dz < img_depth * c.
            Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If ((a, b), (c, d), (e, f)), the scale is randomly sampled from the range a <= scale_x <= b,
            c <= scale_y <= d, e <= scale_z <= f. Will keep original scale by default.
        shears: Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be applied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
          to the batch form (False). Default: False.

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
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomAffine3D((15., 20., 20.), p=1.)
        >>> aug(input), aug.transform_matrix
        (tensor([[[[[0.4503, 0.4763, 0.1680],
                   [0.2029, 0.4267, 0.3515],
                   [0.3195, 0.5436, 0.3706]],
        <BLANKLINE>
                  [[0.5255, 0.3508, 0.4858],
                   [0.0795, 0.1689, 0.4220],
                   [0.5306, 0.7234, 0.6879]],
        <BLANKLINE>
                  [[0.2971, 0.2746, 0.3471],
                   [0.4924, 0.4960, 0.6460],
                   [0.3187, 0.4556, 0.7596]]]]]), tensor([[[ 0.9722, -0.0603,  0.2262, -0.1381],
                 [ 0.1131,  0.9669, -0.2286,  0.1486],
                 [-0.2049,  0.2478,  0.9469,  0.0102],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]))

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32, 32)
        >>> aug = RandomAffine3D((15., 20., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        degrees: Union[
            Tensor,
            float,
            Tuple[float, float],
            Tuple[float, float, float],
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ],
        translate: Optional[Union[Tensor, Tuple[float, float, float]]] = None,
        scale: Optional[
            Union[Tensor, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]
        ] = None,
        shears: Union[
            None,
            Tensor,
            float,
            Tuple[float, float],
            Tuple[float, float, float, float, float, float],
            Tuple[
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
                Tuple[float, float],
            ],
        ] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.degrees = degrees
        self.shears = shears
        self.translate = translate
        self.scale = scale

        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}
        self._param_generator = rg.AffineGenerator3D(degrees, translate, scale, shears)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        transform: Tensor = get_affine_matrix3d(
            params["translations"],
            params["center"],
            params["scale"],
            params["angles"],
            deg2rad(params["sxy"]),
            deg2rad(params["sxz"]),
            deg2rad(params["syx"]),
            deg2rad(params["syz"]),
            deg2rad(params["szx"]),
            deg2rad(params["szy"]),
        ).to(input)
        return transform

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the transform to be a Tensor. Gotcha {type(transform)}")

        return warp_affine3d(
            input,
            transform[:, :3, :],
            (input.shape[-3], input.shape[-2], input.shape[-1]),
            flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
        )
