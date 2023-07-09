from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.transform import elastic_transform2d


class RandomElasticTransform(AugmentationBase2D):
    r"""Add random elastic transformation to a tensor image.

    .. image:: _static/img/RandomElasticTransform.png

    Args:
        kernel_size: the size of the Gaussian kernel.
        sigma: The standard deviation of the Gaussian in the y and x directions,
          respectively. Larger sigma results in smaller pixel displacements.
        alpha: The scaling factor that controls the intensity of the deformation
          in the y and x directions, respectively.
        align_corners: Interpolation flag used by `grid_sample`.
        resample: Interpolation mode used by `grid_sample`. Either 'nearest' (0) or 'bilinear' (1).
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
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        padding_mode: str = "zeros",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        self.flags = {
            "kernel_size": kernel_size,
            "sigma": sigma,
            "alpha": alpha,
            "align_corners": align_corners,
            "resample": Resample.get(resample),
            "padding_mode": padding_mode,
        }

    def generate_parameters(self, shape: Tuple[int, ...]) -> Dict[str, Tensor]:
        B, _, H, W = shape
        if self.same_on_batch:
            noise = torch.rand(1, 2, H, W, device=self.device, dtype=self.dtype).expand(B, 2, H, W)
        else:
            noise = torch.rand(B, 2, H, W, device=self.device, dtype=self.dtype)
        return {"noise": noise * 2 - 1}

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return elastic_transform2d(
            input,
            params["noise"].to(input),
            flags["kernel_size"],
            flags["sigma"],
            flags["alpha"],
            flags["align_corners"],
            flags["resample"].name.lower(),
            flags["padding_mode"],
        )

    def apply_non_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        return input

    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process masks corresponding to the inputs that are transformed."""
        return self.apply_transform(input, params=params, flags=flags, transform=transform)

    def apply_transform_box(
        self, input: Boxes, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Boxes:
        """Process masks corresponding to the inputs that are transformed."""
        # We assume that boxes may not be affected too much by the deformation.
        return input

    def apply_transform_class(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        """Process class tags corresponding to the inputs that are transformed."""
        return input
