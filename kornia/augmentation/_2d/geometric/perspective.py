from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor, as_tensor
from kornia.geometry.transform import get_perspective_transform, warp_perspective


class RandomPerspective(GeometricAugmentationBase2D):
    r"""Apply a random perspective transformation to an image tensor with a given probability.

    .. image:: _static/img/RandomPerspective.png

    Args:
        distortion_scale: the degree of distortion, ranged from 0 to 1.
        resample: the interpolation method to use.
        same_on_batch: apply the same transformation across the batch. Default: False.
        align_corners: interpolation flag.
        p: probability of the image being perspectively transformed.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
        sampling_method: ``'basic'`` | ``'area_preserving'``. Default: ``'basic'``
            If ``'basic'``, samples by translating the image corners randomly inwards.
            If ``'area_preserving'``, samples by randomly translating the image corners in any direction.
            Preserves area on average. See https://arxiv.org/abs/2104.03308 for further details.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.warp_pespective`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs= torch.tensor([[[[1., 0., 0.],
        ...                         [0., 1., 0.],
        ...                         [0., 0., 1.]]]])
        >>> aug = RandomPerspective(0.5, p=0.5)
        >>> out = aug(inputs)
        >>> out
        tensor([[[[0.0000, 0.2289, 0.0000],
                  [0.0000, 0.4800, 0.0000],
                  [0.0000, 0.0000, 0.0000]]]])
        >>> aug.inverse(out)
        tensor([[[[0.0500, 0.0961, 0.0000],
                  [0.2011, 0.3144, 0.0000],
                  [0.0031, 0.0130, 0.0053]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomPerspective(0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        distortion_scale: Union[Tensor, float] = 0.5,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        sampling_method: str = "basic",
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PerspectiveGenerator(distortion_scale, sampling_method=sampling_method)

        self.flags: Dict[str, Any] = {"align_corners": align_corners, "resample": Resample.get(resample)}

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        return get_perspective_transform(params["start_points"].to(input), params["end_points"].to(input))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, _, height, width = input.shape
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return warp_perspective(
            input, transform, (height, width), mode=flags["resample"].name.lower(), align_corners=flags["align_corners"]
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
