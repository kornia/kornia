from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.core import Tensor, as_tensor
from kornia.geometry.conversions import deg2rad
from kornia.geometry.transform import get_affine_matrix2d, warp_affine


class RandomAffine(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a tensor image.

    .. image:: _static/img/RandomAffine.png

    The transformation is computed so that the image center is kept invariant.

    Args:
        degrees: Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale: scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If (a, b, c, d), the scale is randomly sampled from the range a <= scale_x <= b, c <= scale_y <= d.
            Will keep original scale by default.
        shear: Range of degrees to select from.
            If float, a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b), a shear parallel to the x axis in the range (-shear, +shear) will be applied.
            If (a, b, c, d), then x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3])
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "zeros" (0), "border" (1) or "reflection" (2).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.3961, 0.7310, 0.1574],
                  [0.1781, 0.3074, 0.5648],
                  [0.4804, 0.8379, 0.4234]]]]), tensor([[[ 0.9923, -0.1241,  0.1319],
                 [ 0.1241,  0.9923, -0.1164],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.3890, 0.6573, 0.1865],
                  [0.2063, 0.3074, 0.5459],
                  [0.3892, 0.7896, 0.4224]]]])
        >>> input
        tensor([[[[0.4963, 0.7682, 0.0885],
                  [0.1320, 0.3074, 0.6341],
                  [0.4901, 0.8964, 0.4556]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomAffine((-15., 20.), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float]],
        translate: Optional[Union[Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[Tensor, float, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: rg.AffineGenerator = rg.AffineGenerator(degrees, translate, scale, shear)
        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
        }

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return get_affine_matrix2d(
            as_tensor(params["translations"], device=input.device, dtype=input.dtype),
            as_tensor(params["center"], device=input.device, dtype=input.dtype),
            as_tensor(params["scale"], device=input.device, dtype=input.dtype),
            as_tensor(params["angle"], device=input.device, dtype=input.dtype),
            deg2rad(as_tensor(params["shear_x"], device=input.device, dtype=input.dtype)),
            deg2rad(as_tensor(params["shear_y"], device=input.device, dtype=input.dtype)),
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return warp_affine(
            input,
            transform[:, :2, :],
            (height, width),
            flags["resample"].name.lower(),
            align_corners=flags["align_corners"],
            padding_mode=flags["padding_mode"].name.lower(),
        )

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
