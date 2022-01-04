from typing import Dict, Optional, cast

import torch

from kornia.augmentation.base_2d import GeometricAugmentationBase2D
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid

from .. import random_generator as rg


class RandomFisheye(GeometricAugmentationBase2D):
    r"""Add random camera radial distortion.

    .. image:: _static/img/RandomFisheye.png

    Args:
        center_x: Ranges to sample respect to x-coordinate center with shape (2,).
        center_y: Ranges to sample respect to y-coordinate center with shape (2,).
        gamma: Ranges to sample for the gamma values respect to optical center with shape (2,).
        return_transform: if ``True`` return the matrix describing the transformation applied to each
            input tensor. If ``False`` and the input is a tuple the applied transformation won't be concatenated.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Examples:
        >>> img = torch.ones(1, 1, 2, 2)
        >>> center_x = torch.tensor([-.3, .3])
        >>> center_y = torch.tensor([-.3, .3])
        >>> gamma = torch.tensor([.9, 1.])
        >>> out = RandomFisheye(center_x, center_y, gamma)(img)
        >>> out.shape
        torch.Size([1, 1, 2, 2])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomFisheye(center_x, center_y, gamma, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        gamma: torch.Tensor,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )
        self._check_tensor(center_x)
        self._check_tensor(center_y)
        self._check_tensor(gamma)
        self._param_generator = cast(
            rg.PlainUniformGenerator,
            rg.PlainUniformGenerator(
                (center_x[:, None], "center_x", None, None),
                (center_y[:, None], "center_y", None, None),
                (gamma[:, None], "gamma", None, None),
            ),
        )

    def _check_tensor(self, data: torch.Tensor) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Invalid input type. Expected torch.Tensor - got: {type(data)}")

        if len(data.shape) != 1 and data.shape[0] != 2:
            raise ValueError(f"Tensor must be of shape (2,). Got: {data.shape}.")

    # TODO: It is incorrect to return identity
    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.identity_matrix(input)

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # create the initial sampling fields
        B, _, H, W = input.shape
        grid = create_meshgrid(H, W, normalized_coordinates=True)
        field_x = grid[..., 0].to(input)  # 1xHxW
        field_y = grid[..., 1].to(input)  # 1xHxW
        # vectorize the random parameters
        center_x = params["center_x"].view(B, 1, 1).to(input)
        center_y = params["center_y"].view(B, 1, 1).to(input)
        gamma = params["gamma"].view(B, 1, 1).to(input)
        # compute and apply the distances respect to the camera optical center
        distance = ((center_x - field_x) ** 2 + (center_y - field_y) ** 2) ** 0.5
        field_x = field_x + field_x * distance ** gamma  # BxHxw
        field_y = field_y + field_y * distance ** gamma  # BxHxW
        return remap(input, field_x, field_y, normalized_coordinates=True, align_corners=True)
