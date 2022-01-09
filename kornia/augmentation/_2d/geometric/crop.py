from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from torch.nn.functional import pad

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation.base import TensorWithTransformMat
from kornia.augmentation.utils import _transform_input
from kornia.constants import Resample
from kornia.geometry.transform import crop_by_transform_mat, get_perspective_transform


class RandomCrop(GeometricAugmentationBase2D):
    r"""Crop random patches of a tensor image on a given size.

    .. image:: _static/img/RandomCrop.png

    Args:
        p: probability of applying the transformation for the whole batch.
        size: Desired output size (out_h, out_w) of the crop.
            Must be Tuple[int, int], then out_h = size[0], out_w = size[1].
        padding: Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed: It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric.
        resample: the interpolation mode.
        return_transform: if ``True`` return the matrix describing the transformation applied to each
                          input tensor. If ``False`` and the input is a tuple the applied transformation
                          won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
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

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> inputs = torch.arange(1*1*3*3.).view(1, 1, 3, 3)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> out = aug(inputs)
        >>> out
        tensor([[[[3., 4.],
                  [6., 7.]]]])
        >>> aug.inverse(out, padding_mode="border")
        tensor([[[[3., 4., 4.],
                  [3., 4., 4.],
                  [6., 7., 7.]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomCrop((2, 2), p=1., cropping_mode="resample")
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        size: Tuple[int, int],
        padding: Optional[Union[int, Tuple[int, int], Tuple[int, int, int, int]]] = None,
        pad_if_needed: Optional[bool] = False,
        fill: int = 0,
        padding_mode: str = "constant",
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 1.0,
        keepdim: bool = False,
        cropping_mode: str = "slice",
    ) -> None:
        # Since PyTorch does not support ragged tensor. So cropping function happens batch-wisely.
        super().__init__(
            p=1.0, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=p, keepdim=keepdim
        )
        self._param_generator = cast(rg.CropGenerator, rg.CropGenerator(size))
        self.flags = dict(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
            resample=Resample.get(resample),
            align_corners=align_corners,
            cropping_mode=cropping_mode,
        )

    def compute_padding(self, shape: torch.Size) -> List[int]:
        if len(shape) != 4:
            raise AssertionError(f"Expected BCHW. Got {shape}.")
        padding = [0, 0, 0, 0]
        if self.flags["padding"] is not None:
            if isinstance(self.flags["padding"], int):
                padding = [self.flags["padding"]] * 4
            elif isinstance(self.flags["padding"], tuple) and len(self.flags["padding"]) == 2:
                padding = [
                    self.flags["padding"][1],
                    self.flags["padding"][1],
                    self.flags["padding"][0],
                    self.flags["padding"][0],
                ]
            elif isinstance(self.flags["padding"], tuple) and len(self.flags["padding"]) == 4:
                padding = [
                    self.flags["padding"][3],
                    self.flags["padding"][2],
                    self.flags["padding"][1],
                    self.flags["padding"][0],
                ]
            else:
                raise RuntimeError(f"Expect `padding` to be a scalar, or length 2/4 list. Got {self.flags['padding']}.")

        if self.flags["pad_if_needed"] and shape[-2] < self.flags["size"][0]:
            padding = [0, 0, (self.flags["size"][0] - shape[-2]), self.flags["size"][0] - shape[-2]]

        if self.flags["pad_if_needed"] and shape[-1] < self.flags["size"][1]:
            padding = [self.flags["size"][1] - shape[-1], self.flags["size"][1] - shape[-1], 0, 0]

        return padding

    def precrop_padding(self, input: torch.Tensor, padding: List[int] = None) -> torch.Tensor:
        if padding is None:
            padding = self.compute_padding(input.shape)

        input = pad(input, padding, value=self.flags["fill"], mode=self.flags["padding_mode"])

        return input

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        transform: torch.Tensor = get_perspective_transform(params["src"].to(input), params["dst"].to(input))
        return transform

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.flags["cropping_mode"] == "resample":  # uses bilinear interpolation to crop
            transform = cast(torch.Tensor, transform)
            return crop_by_transform_mat(
                input,
                transform,
                self.flags["size"],
                mode=self.flags["resample"].name.lower(),
                padding_mode="zeros",
                align_corners=self.flags["align_corners"],
            )
        if self.flags["cropping_mode"] == "slice":  # uses advanced slicing to crop
            B, C, _, _ = input.shape
            out = torch.empty(B, C, *self.flags["size"], device=input.device, dtype=input.dtype)
            for i in range(B):
                x1 = int(params["src"][i, 0, 0])
                x2 = int(params["src"][i, 1, 0]) + 1
                y1 = int(params["src"][i, 0, 1])
                y2 = int(params["src"][i, 3, 1]) + 1
                out[i] = input[i : i + 1, :, y1:y2, x1:x2]
            return out
        raise NotImplementedError(f"Not supported type: {self.flags['cropping_mode']}.")

    def inverse_transform(
        self,
        input: torch.Tensor,
        transform: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.flags["cropping_mode"] != "resample":
            raise NotImplementedError(
                f"`inverse` is only applicable for resample cropping mode. Got {self.flags['cropping_mode']}."
            )
        size = cast(Tuple[int, int], size)
        mode = self.flags["resample"].name.lower() if "mode" not in kwargs else kwargs["mode"]
        align_corners = self.flags["align_corners"] if "align_corners" not in kwargs else kwargs["align_corners"]
        padding_mode = "zeros" if "padding_mode" not in kwargs else kwargs["padding_mode"]
        transform = cast(torch.Tensor, transform)
        return crop_by_transform_mat(input, transform[:, :2, :], size, mode, padding_mode, align_corners)

    def inverse(
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        out = super().inverse(input, params, size, **kwargs)
        if params is None:
            params = self._params
        if "padding_size" in params:
            padding_size = params["padding_size"].unique(dim=0).cpu().squeeze().numpy().tolist()
            padding_size = [-padding_size[0], -padding_size[1], -padding_size[2], -padding_size[3]]
        else:
            padding_size = [0, 0, 0, 0]
        return self.precrop_padding(out, padding_size)

    def forward_parameters_precrop(self, batch_shape) -> Dict[str, torch.Tensor]:
        input_pad = self.compute_padding(batch_shape)
        batch_shape_new = (
            *batch_shape[:2],
            batch_shape[2] + input_pad[2] + input_pad[3],  # original height + top + bottom padding
            batch_shape[3] + input_pad[0] + input_pad[1],  # original width + left + right padding
        )
        padding_size = torch.tensor(tuple(input_pad), dtype=torch.long).expand(batch_shape[0], -1)
        _params = super().forward_parameters(batch_shape_new)
        _params.update({"padding_size": padding_size})
        return _params

    def forward(
        self,
        input: TensorWithTransformMat,
        params: Optional[Dict[str, torch.Tensor]] = None,
        return_transform: Optional[bool] = None,
    ) -> TensorWithTransformMat:
        padding_size = params.get("padding_size") if params else None
        if padding_size is not None:
            input_pad = padding_size.unique(dim=0).cpu().squeeze().numpy().tolist()
        else:
            input_pad = None

        if isinstance(input, (tuple, list)):
            input_temp = _transform_input(input[0])
            input_pad = self.compute_padding(input[0].shape) if input_pad is None else input_pad
            _input = (self.precrop_padding(input_temp, input_pad), input[1])
        else:
            input = cast(torch.Tensor, input)  # TODO: weird that cast is not working under this context.
            input_temp = _transform_input(input)
            input_pad = self.compute_padding(input_temp.shape) if input_pad is None else input_pad
            _input = self.precrop_padding(input_temp, input_pad)  # type: ignore
        out = super().forward(_input, params, return_transform)

        # Update the actual input size for inverse
        if "padding_size" not in self._params:
            _padding_size = torch.tensor(tuple(input_pad), device=input_temp.device, dtype=torch.long).expand(
                input_temp.size(0), -1
            )
            self._params.update({"padding_size": _padding_size})

        if not self._params["batch_prob"].all():
            # undo the pre-crop if nothing happened.
            if isinstance(out, tuple) and isinstance(input, tuple):
                return input[0], out[1]
            if isinstance(out, tuple) and not isinstance(input, tuple):
                return input, out[1]
            return input
        return out
