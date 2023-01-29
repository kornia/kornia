from typing import Any, Dict, Optional, Tuple, Union, cast

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.core import Tensor
from kornia.geometry.transform import get_translation_matrix2d, warp_affine


class RandomTranslate(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a tensor image.

    Args:
        translate: tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            will be applied. Will not apply shear by default.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "zeros" (0), "border" (1) or "refection" (2).
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_affine`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3)
        >>> aug = RandomTranslate((0.2, 0.1), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.0050, 0.0523, 0.0700],
                  [0.0460, 0.4861, 0.6642],
                  [0.0168, 0.1877, 0.3913]]]]), tensor([[[1.0000, 0.0000, 0.6000],
                 [0.0000, 1.0000, 0.6000],
                 [0.0000, 0.0000, 1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.4026, 0.5886, 0.0605],
                  [0.1977, 0.3985, 0.0419],
                  [0.0171, 0.0371, 0.0039]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomTranslate((0.2, 0.1), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        translate: Union[Tensor, Tuple[float, float]],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: rg.TranslateGenerator = rg.TranslateGenerator(translate)
        self.flags = dict(
            resample=Resample.get(resample), padding_mode=SamplePadding.get(padding_mode), align_corners=align_corners
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        translations = torch.stack([params["translate_x"], params["translate_x"]], dim=-1)
        return get_translation_matrix2d(
            torch.as_tensor(translations, device=input.device, dtype=input.dtype),
        )

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        transform = cast(Tensor, transform)

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
        return self.apply_transform(
            input,
            params=self._params,
            transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
