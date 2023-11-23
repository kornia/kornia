import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.nerf.samplers import calc_ray_t_vals


class VolumeRenderer(torch.nn.Module):
    r"""Base class for volume rendering.

    Implementation follows Ben Mildenhall et el. (2020) at https://arxiv.org/abs/2003.08934.
    """

    _huge = 1.0e10
    _eps = 1.0e-10

    def __init__(self, shift: int = 1) -> None:
        """Initializes the renderer.

        Args:
            shift: Size of far-field layer: int
        """
        super().__init__()
        self._shift = shift

    def _render(self, alpha: Tensor, rgbs: Tensor) -> Tensor:
        trans = torch.cumprod(1 - alpha + self._eps, dim=-2)  # (*, N, 1)
        trans = torch.roll(trans, shifts=self._shift, dims=-2)  # (*, N, 1)
        trans[..., : self._shift, :] = 1  # (*, N, 1)

        weights = trans * alpha  # (*, N, 1)

        rgbs_rendered = torch.sum(weights * rgbs, dim=-2)  # (*, 3)

        return rgbs_rendered

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tensor:
        raise NotImplementedError


class IrregularRenderer(VolumeRenderer):
    """Renders 3D irregularly sampled points along rays."""

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tensor:
        r"""Renders 3D irregularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            points_3d: 3D points along rays :math:`(*, N, 3)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`
        """
        t_vals = calc_ray_t_vals(points_3d)
        deltas = t_vals[..., 1:] - t_vals[..., :-1]  # (*, N - 1)
        far = torch.empty(size=t_vals.shape[:-1], dtype=t_vals.dtype, device=t_vals.device).fill_(self._huge)
        deltas = torch.cat([deltas, far[..., None]], dim=-1)  # (*, N)

        alpha = 1 - torch.exp(-1.0 * densities * deltas[..., None])  # (*, N)

        return self._render(alpha, rgbs)


class RegularRenderer(VolumeRenderer):
    """Renders 3D regularly sampled points along rays."""

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tensor:
        r"""Renders 3D regularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            points_3d: 3D points along rays :math:`(*, N, 3)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`
        """
        KORNIA_CHECK_SHAPE(rgbs, ["*", "N", "3"])
        KORNIA_CHECK_SHAPE(densities, ["*", "N"])
        KORNIA_CHECK_SHAPE(points_3d, ["*", "N", "3"])

        num_ray_points: int = points_3d.shape[-2]

        points_3d = points_3d.reshape(-1, num_ray_points, 3)  # (*, N, 3)

        delta_3d = points_3d[0, 1, :] - points_3d[0, 0, :]  # (*, 3)
        delta = torch.linalg.norm(delta_3d, dim=-1)

        alpha = 1 - torch.exp(-1.0 * densities * delta)  # (*, N)

        return self._render(alpha, rgbs)
