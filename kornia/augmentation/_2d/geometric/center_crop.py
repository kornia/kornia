from typing import Any, Dict, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.core import Tensor
from kornia.constants import Resample
from kornia.geometry.transform import crop_by_indices, crop_by_transform_mat, get_perspective_transform
from kornia.geometry.bbox import infer_bbox_shape


class CenterCrop(GeometricAugmentationBase2D):
    r"""Crop a given image tensor at the center.

    .. image:: _static/img/CenterCrop.png

    Args:
        size: Desired output size (out_h, out_w) of the crop.
            If integer,  out_h = out_w = size.
            If Tuple[int, int], out_h = size[0], out_w = size[1].
        align_corners: interpolation flag.
        resample: The interpolation mode.
        p: probability of applying the transformation for the whole batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).
        cropping_mode: The used algorithm to crop. ``slice`` will use advanced slicing to extract the tensor based
                       on the sampled indices. ``resample`` will use `warp_affine` using the affine transformation
                       to extract and resize at once. Use `slice` for efficiency, or `resample` for proper
                       differentiability.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, out_h, out_w)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.crop_by_boxes`.

    Examples:
        >>> import torch
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn(1, 1, 4, 4)
        >>> inputs
        tensor([[[[-1.1258, -1.1524, -0.2506, -0.4339],
                  [ 0.8487,  0.6920, -0.3160, -2.1152],
                  [ 0.3223, -1.2633,  0.3500,  0.3081],
                  [ 0.1198,  1.2377,  1.1168, -0.2473]]]])
        >>> aug = CenterCrop(2, p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[ 0.6920, -0.3160],
                  [-1.2633,  0.3500]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[ 0.6920,  0.6920, -0.3160, -0.3160],
                  [ 0.6920,  0.6920, -0.3160, -0.3160],
                  [-1.2633, -1.2633,  0.3500,  0.3500],
                  [-1.2633, -1.2633,  0.3500,  0.3500]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = CenterCrop(2, p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        align_corners: bool = True,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # same_on_batch is always True for CenterCrop
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super().__init__(p=1.0, same_on_batch=True, p_batch=p, keepdim=keepdim)
        if isinstance(size, tuple):
            self.size = (size[0], size[1])
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise Exception(f"Invalid size type. Expected (int, tuple(int, int). " f"Got: {type(size)}.")

        self._param_generator = rg.CenterCropGenerator(self.size)
        self.flags = dict(
            resample=Resample.get(resample),
            cropping_mode=cropping_mode,
            align_corners=align_corners,
            size=self.size,
            padding_mode="zeros",
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] in ("resample", "slice"):
            transform = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
            transform = transform.expand(input.shape[0], -1, -1)
            return transform
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            transform = self.get_transformation_matrix(input, params=params, flags=flags)

            return crop_by_transform_mat(
                input, transform[:, :2, :], self.size, flags["resample"].name.lower(), "zeros", flags["align_corners"]
            )
        if flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            height, width = infer_bbox_shape(params["dst"])
            height = height.unique(sorted=False)
            width = width.unique(sorted=False)
            if not (len(height) == len(width) == 1):
                raise RuntimeError(f"Invalid dst boxes with multiple height {height} and width {width}.")
            return crop_by_indices(
                input, params["src"].to(device=input.device), (height.long().item(), width.long().item()))
        raise NotImplementedError(f"Not supported type: {flags['cropping_mode']}.")

    def inverse_transform(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        if flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {flags['cropping_mode']}."
            )
        size = params['forward_input_shape'].numpy().tolist()
        size = (size[-2], size[-1])
        transform = self.get_inverse_transformation_matrix(input, params=params, flags=flags)

        return crop_by_transform_mat(
            input,
            transform[:, :2, :],
            size,
            flags["resample"].name.lower(),
            flags["padding_mode"],
            flags["align_corners"],
        )
