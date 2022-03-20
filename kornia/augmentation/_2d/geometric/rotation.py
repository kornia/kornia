from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.geometry.transform import affine
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center


class RandomRotation(GeometricAugmentationBase2D):
    r"""Apply a random rotation to a tensor image or a batch of tensor images given an amount of degrees.

    .. image:: _static/img/RandomRotation.png

    Args:
        degrees: range of degrees to select from. If degrees is a number the
          range of degrees to select from will be (-degrees, +degrees).
        resample: Default: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.affine`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.tensor([[1., 0., 0., 2.],
        ...                       [0., 0., 0., 0.],
        ...                       [0., 1., 2., 0.],
        ...                       [0., 0., 1., 2.]])
        >>> aug = RandomRotation(degrees=45.0, p=1.)
        >>> out = aug(input)
        >>> out
        tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                  [0.0000, 0.0029, 0.0000, 0.0176],
                  [0.0029, 1.0000, 1.9883, 0.0000],
                  [0.0000, 0.0088, 1.0117, 1.9649]]]])
        >>> aug.transform_matrix
        tensor([[[ 1.0000, -0.0059,  0.0088],
                 [ 0.0059,  1.0000, -0.0088],
                 [ 0.0000,  0.0000,  1.0000]]])
        >>> aug.inverse(out)
        tensor([[[[9.6526e-01, 8.6824e-03, 1.7263e-02, 1.9305e+00],
                  [8.6398e-03, 2.9485e-03, 5.8971e-03, 1.7365e-02],
                  [2.9054e-03, 9.9416e-01, 1.9825e+00, 2.3134e-02],
                  [2.5777e-05, 1.1640e-02, 9.9992e-01, 1.9392e+00]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomRotation(degrees=45.0, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    # Note: Extra params, center=None, fill=0 in TorchVision

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float], List[float]],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = cast(
            rg.PlainUniformGenerator, rg.PlainUniformGenerator((degrees, "degrees", 0.0, (-360.0, 360.0)))
        )
        self.flags = dict(resample=Resample.get(resample), align_corners=align_corners)

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        # TODO: Update to use `get_rotation_matrix2d`
        angles: Tensor = params["degrees"].to(input)

        center: Tensor = _compute_tensor_center(input)
        rotation_mat: Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: Tensor = torch.eye(3, device=input.device, dtype=input.dtype).repeat(input.shape[0], 1, 1)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]

        return trans_mat

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        transform = cast(Tensor, transform)

        return affine(
            input, transform[..., :2, :3], flags["resample"].name.lower(), "zeros", flags["align_corners"]
        )

    def inverse_transform(
        self,
        input: Tensor,
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        return self.apply_transform(
            input, params=self._params, transform=torch.as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags
        )
