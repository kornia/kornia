import torch

from kornia.core import Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.ray import Ray
from kornia.nerf.nerf_model import NerfModel
from kornia.nerf.rays import calc_ray_t_vals
from kornia.utils._compat import torch_inference_mode
from kornia.utils.grid import create_meshgrid


class VolumeRenderer(torch.nn.Module):
    r"""Volume renderer class. Implementation follows Ben Mildenhall et el. (2020) at
    https://arxiv.org/abs/2003.08934.

    Args:
        shift: Size of far-field layer: int
    """

    _huge = 1.0e10
    _eps = 1.0e-10

    def __init__(self, shift: int = 1) -> None:
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
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tensor:
        r"""Renders 3D irregularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            t_vals: Sampled point distances along rays :math: `(*, N)`

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
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tensor:
        r"""Renders 3D regularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            deltas: Equispaced distance for point pairs along each ray :math: `(*)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`
        """
        num_ray_points = points_3d.shape[-2]
        delta_3d = points_3d.reshape(-1, num_ray_points, 3)[0, 1, :] - points_3d.reshape(-1, num_ray_points, 3)[0, 0, :]
        delta = torch.linalg.norm(delta_3d, dim=-1)

        alpha = 1 - torch.exp(-1.0 * densities * delta)  # (*, N)

        return self._render(alpha, rgbs)


class NerfModelRenderer:
    r"""Renders a novel synthesis view of a trained NeRF model for given camera.

    Args:
        nerf_model: NeRF model: NerfModel
        image_size: image size: tuple[int, int]
    """

    def __init__(
        self, nerf_model: NerfModel, image_size: tuple[int, int], device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        self._nerf_model = nerf_model
        self._image_size = image_size
        self._device = device
        self._dtype = dtype

        self._pixels_grid, self._ones = self._create_pixels_grid()  # 1xHxWx2 and (H*W)x1

    def _create_pixels_grid(self) -> Tensor:
        r"""Creates the pixels grid to unproject to plane z=1.

        Args:
            image_size: image size: tuple[int, int]

        Returns:
            Pixels grid: Tensor (1, H, W, 2) and (H*W, 1).
        """
        height, width = self._image_size
        pixels_grid = create_meshgrid(
            height, width, normalized_coordinates=False, device=self._device, dtype=self._dtype
        )  # 1xHxWx2
        pixels_grid = pixels_grid.reshape(-1, 2)  # (H*W)x2

        ones = torch.ones(pixels_grid.shape[0], 1, device=pixels_grid.device, dtype=pixels_grid.dtype)  # (H*W)x1

        return pixels_grid, ones

    def render_view(self, camera: PinholeCamera) -> Tensor:
        r"""Renders a novel synthesis view of a trained NeRF model for given camera.

        Args:
            camera: camera for image rendering: PinholeCamera.

        Returns:
            Rendered image: Tensor (H, W, C).
        """
        # create ray for this camera
        rays: Ray = self._create_rays(camera)

        # render the image
        with torch_inference_mode():
            rgb_model = self._nerf_model(rays.origins, rays.directions)

        rgb_image = rgb_model.view(self._image_size[0], self._image_size[1], 3)

        return rgb_image

    def _create_rays(self, camera: PinholeCamera) -> Ray:
        """Creates rays for a given camera.

        Args:
            camera: camera for image rendering: PinholeCamera.
        """
        height, width = self._image_size

        # convert to rays
        origin = camera.extrinsics[:, :3, -1]  # 1x3
        origin = origin.repeat(height * width, 1)  # (H*W)x3

        destination = camera.unproject(self._pixels_grid, self._ones)  # (H*W)x3

        return Ray.through(origin, destination)
