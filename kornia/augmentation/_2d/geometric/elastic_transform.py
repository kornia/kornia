from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.geometry.transform import elastic_transform2d


class RandomElasticTransform(GeometricAugmentationBase2D):
    r"""Add random elastic transformation to a tensor image.

    .. image:: _static/img/RandomElasticTransform.png

    Args:
        kernel_size: the size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian in the y and x directions,
          respectively. Larger sigma results in smaller pixel displacements.
        alpha: The scaling factor that controls the intensity of the deformation
          in the y and x directions, respectively.
        align_corners: Interpolation flag used by `grid_sample`.
        mode: Interpolation mode used by `grid_sample`. Either 'bilinear' or 'nearest'.
        padding_mode: The padding used by ```grid_sample```. Either 'zeros', 'border' or 'refection'.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False).

    .. note::
        This function internally uses :func:`kornia.geometry.transform.elastic_transform2d`.

    Examples:
        >>> import torch
        >>> img = torch.ones(1, 1, 2, 2)
        >>> out = RandomElasticTransform()(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomElasticTransform(p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (63, 63),
        sigma: Tuple[float, float] = (32.0, 32.0),
        alpha: Tuple[float, float] = (1.0, 1.0),
        align_corners: bool = False,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self.flags = dict(
            kernel_size=kernel_size,
            sigma=sigma,
            alpha=alpha,
            align_corners=align_corners,
            mode=mode,
            padding_mode=padding_mode,
        )

    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        B, _, H, W = shape
        if self.same_on_batch:
            noise = torch.rand(1, 2, H, W, device=self.device, dtype=self.dtype).repeat(B, 1, 1, 1)
        else:
            noise = torch.rand(B, 2, H, W, device=self.device, dtype=self.dtype)
        return dict(noise=noise * 2 - 1)

    # TODO: It is incorrect to return identity
    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor]) -> Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        return elastic_transform2d(
            input,
            params["noise"].to(input),
            self.flags["kernel_size"],
            self.flags["sigma"],
            self.flags["alpha"],
            self.flags["align_corners"],
            self.flags["mode"],
            self.flags["padding_mode"],
        )
