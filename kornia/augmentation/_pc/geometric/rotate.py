from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._pc.geometric.base import GeometricAugmentationBasePC
from kornia.core import Tensor, as_tensor
from kornia.geometry.conversions import angle_to_rotation_matrix
from kornia.utils.misc import eye_like


class RandomRotatePC(GeometricAugmentationBasePC):
    r"""Rotates point clouds.

    Args:
        degrees: range of degrees to select from. If degrees is a number the
          range of degrees to select from will be (-degrees, +degrees).
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    """

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float], List[float]],
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((degrees, "degrees", 0.0, (-360.0, 360.0)))

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        angles: Tensor = params["degrees"].to(input)

        rotation_mat: Tensor = angle_to_rotation_matrix(angles)

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: Tensor = eye_like(3, input, shared_memory=False)
        trans_mat[:, :2, :2] = rotation_mat

        return trans_mat

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the `transform` be a Tensor. Got {type(transform)}.')

        return input[..., :2] * transform[:, :2, :2]

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the `transform` be a Tensor. Got {type(transform)}.')
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
