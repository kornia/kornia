from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample, SamplePadding
from kornia.core import Tensor, as_tensor, stack
from kornia.geometry.transform import get_translation_matrix2d, warp_affine


class RandomTranslate(GeometricAugmentationBase2D):
    r"""Apply a random 2D affine transformation to a tensor image.

    Args:
        translate_x: tuple of maximum absolute fraction for horizontal translations.
            For example translate_x=(a, b), then horizontal shift
            is randomly sampled in the range img_width * a < dx < img_width * b
        translate_y: tuple of maximum absolute fraction for vertical translations.
            For example translate_y=(a, b), then vertical shift
            is randomly sampled in the range img_height * a < dy < img_height * b.
        resample: resample mode from "nearest" (0) or "bilinear" (1).
        padding_mode: padding mode from "zeros" (0), "border" (1) or "reflection" (2).
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
        >>> aug = RandomTranslate((-0.2, 0.2), (-0.1, 0.1), p=1.)
        >>> out = aug(input)
        >>> out, aug.transform_matrix
        (tensor([[[[0.3403, 0.6439, 0.2920],
                  [0.1377, 0.3383, 0.5569],
                  [0.3226, 0.6909, 0.4844]]]]), tensor([[[ 1.0000,  0.0000,  0.1588],
                 [ 0.0000,  1.0000, -0.0907],
                 [ 0.0000,  0.0000,  1.0000]]]))
        >>> aug.inverse(out)
        tensor([[[[0.3565, 0.4839, 0.1922],
                  [0.2164, 0.4134, 0.3968],
                  [0.3797, 0.6075, 0.3765]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomTranslate((-0.2, 0.2), (-0.1, 0.1), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        translate_x: Optional[Union[Tensor, Tuple[float, float]]] = None,
        translate_y: Optional[Union[Tensor, Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        padding_mode: Union[str, int, SamplePadding] = SamplePadding.ZEROS.name,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator: rg.TranslateGenerator = rg.TranslateGenerator(translate_x, translate_y)
        self.flags = {
            "resample": Resample.get(resample),
            "padding_mode": SamplePadding.get(padding_mode),
            "align_corners": align_corners,
        }

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        translations = stack([params["translate_x"], params["translate_y"]], dim=-1)
        return get_translation_matrix2d(as_tensor(translations, device=input.device, dtype=input.dtype))

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
