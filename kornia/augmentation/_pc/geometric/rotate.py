from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._pc.geometric.base import GeometricAugmentationBasePC
from kornia.core import Tensor, as_tensor
from kornia.geometry.transform.imgwarp import deg2rad, angle_axis_to_rotation_matrix


class RandomRotatePC(GeometricAugmentationBasePC):
    r"""Rotates point clouds.

    Args:
        degrees: range of degrees to select from, contains [(x_min, x_max), (y_min, y_max), (z_min, z_max)].
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    """

    def __init__(
        self,
        degrees: Union[Tensor, Tuple[float, float], List[Tuple[float, float]]] = [(-15., 15.), (-15., 15.), (0., 0.)],
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator(
            (degrees[0], "degrees_x", 0.0, (-360.0, 360.0)),
            (degrees[1], "degrees_y", 0.0, (-360.0, 360.0)),
            (degrees[2], "degrees_z", 0.0, (-360.0, 360.0)),
        )

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        angles: Tensor = torch.stack([params["degrees_x"], params["degrees_y"], params["degrees_z"]], dim=-1).to(input)

        angle_axis_rad: Tensor = deg2rad(angles)
        rotation_mat: Tensor = angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3

        return rotation_mat

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f'Expected the `transform` be a Tensor. Got {type(transform)}.')

        input[..., :3] = input[..., :3] * transform
        return input

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
