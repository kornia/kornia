from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform


class RandomResizedCrop(GeometricAugmentationBase2D):
    r"""Crop random patches in an image tensor and resizes to a given size.

    .. image:: _static/img/RandomResizedCrop.png

    Args:
        size: Desired output size (out_h, out_w) of each edge.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        resample: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of the augmentation been applied.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, out_h, out_w)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Example:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.tensor([[[0., 1., 2.],
        ...                         [3., 4., 5.],
        ...                         [6., 7., 8.]]])
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[1.0000, 1.5000, 2.0000],
                  [4.0000, 4.5000, 5.0000],
                  [7.0000, 7.5000, 8.0000]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[1., 1., 2.],
                  [4., 4., 5.],
                  [7., 7., 8.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Union[Tensor, Tuple[float, float]] = (0.08, 1.0),
        ratio: Union[Tensor, Tuple[float, float]] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens all the time.
        super().__init__(p=1.0, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim)
        self._param_generator = rg.ResizedCropGenerator(size, scale, ratio)
        self.flags = {
            "size": size,
            "resample": Resample.get(resample),
            "align_corners": align_corners,
            "cropping_mode": cropping_mode,
            "padding_mode": "zeros",
        }

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            transform: Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            transform = transform.expand(input.shape[0], -1, -1)
            return transform
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            if not isinstance(transform, Tensor):
                raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

            return crop_by_transform_mat(
                input,
                transform,
                flags["size"],
                mode=flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=flags["align_corners"],
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            return crop_by_indices(
                input,
                params["src"],
                flags["size"],
                interpolation=flags["resample"].name.lower(),
                align_corners=flags["align_corners"],
            )
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        if not isinstance(size, tuple):
            raise TypeError(f"Expected the size be a tuple. Gotcha {type(size)}")

        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return crop_by_transform_mat(
            input,
            transform[:, :2, :],
            size,
            flags["resample"].name.lower(),
            flags["padding_mode"],
            flags["align_corners"],
        )
