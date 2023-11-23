from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from kornia.core import Module, Tensor
from kornia.geometry.camera import PinholeCamera
from kornia.geometry.ray import Ray
from kornia.nerf.positional_encoder import PositionalEncoder
from kornia.nerf.samplers import sample_lengths, sample_ray_points
from kornia.nerf.volume_renderer import IrregularRenderer, RegularRenderer
from kornia.utils._compat import torch_inference_mode
from kornia.utils.grid import create_meshgrid


class MLP(Module):
    r"""Class to represent a multi-layer perceptron.

    The MLP represents a deep NN of fully connected layers.
    The network is build of user defined sub-units, each with a user defined number of layers.

    Skip connections span between the sub-units.
    The model follows: Ben Mildenhall et el. (2020) at https://arxiv.org/abs/2003.08934.
    """

    def __init__(self, num_dims: int, num_units: int = 2, num_unit_layers: int = 4, num_hidden: int = 128) -> None:
        """Constructor method.

        Args:
            num_dims: Number of input dimensions (channels).
            num_units: Number of sub-units.
            num_unit_layers: Number of fully connected layers in each sub-unit.
            num_hidden: Layer hidden dimensions.
        """
        super().__init__()
        self._num_unit_layers = num_unit_layers
        layers = []
        for i in range(num_units):
            num_unit_inp_dims = num_dims if i == 0 else num_hidden + num_dims
            for j in range(num_unit_layers):
                num_layer_inp_dims = num_unit_inp_dims if j == 0 else num_hidden
                layer = nn.Linear(num_layer_inp_dims, num_hidden)
                layers.append(nn.Sequential(layer, nn.ReLU()))
        self._mlp = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        x_skip = x
        for i, layer in enumerate(self._mlp):
            if i > 0 and i % self._num_unit_layers == 0:
                out = torch.cat((out, x_skip), dim=-1)
            out = layer(out)
        return out


class NerfModel(Module):
    r"""Class to represent NeRF model.

    Args:
        num_ray_points: Number of points to sample along rays.
        irregular_ray_sampling: Whether to sample ray points irregularly.
        num_pos_freqs: Number of frequencies for positional encoding.
        num_dir_freqs: Number of frequencies for directional encoding.
        num_units: Number of sub-units.
        num_unit_layers: Number of fully connected layers in each sub-unit.
        num_hidden: Layer hidden dimensions.
        log_space_encoding: Whether to apply log spacing for encoding.
    """

    def __init__(
        self,
        num_ray_points: int,
        irregular_ray_sampling: bool = True,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        num_units: int = 2,
        num_unit_layers: int = 4,
        num_hidden: int = 128,  # FIXME: add as call argument
        log_space_encoding: bool = True,
    ) -> None:
        super().__init__()
        self._num_ray_points = num_ray_points
        self._irregular_ray_sampling = irregular_ray_sampling
        self._renderer = IrregularRenderer() if self._irregular_ray_sampling else RegularRenderer()

        self._pos_encoder = PositionalEncoder(3, num_pos_freqs, log_space=log_space_encoding)
        self._dir_encoder = PositionalEncoder(3, num_dir_freqs, log_space=log_space_encoding)
        self._mlp = MLP(self._pos_encoder.num_encoded_dims, num_units, num_unit_layers, num_hidden)
        self._fc1 = nn.Linear(num_hidden, num_hidden)
        self._fc2 = nn.Sequential(
            nn.Linear(num_hidden + self._dir_encoder.num_encoded_dims, num_hidden // 2), nn.ReLU()
        )

        self._sigma = nn.Linear(num_hidden, 1, bias=True)
        torch.nn.init.xavier_uniform_(self._sigma.weight.data)
        self._sigma.bias.data = torch.tensor([0.1]).float()

        self._rgb = nn.Sequential(nn.Linear(num_hidden // 2, 3), nn.Sigmoid())
        self._rgb[0].bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, origins: Tensor, directions: Tensor) -> Tensor:
        """Forward method.

        Args:
            origins: Ray origins with shape :math:`(B, 3)`.
            directions: Ray directions with shape :math:`(B, 3)`.

        Returns:
            Rendered image pixels :math:`(B, 3)`.
        """
        # Sample xyz for ray parameters
        batch_size = origins.shape[0]
        lengths = sample_lengths(
            batch_size,
            self._num_ray_points,
            device=origins.device,
            dtype=origins.dtype,
            irregular=self._irregular_ray_sampling,
        )  # FIXME: handle the case of hierarchical sampling
        points_3d = sample_ray_points(origins, directions, lengths)

        # Encode positions & directions
        points_3d_encoded = self._pos_encoder(points_3d)
        directions_encoded = self._dir_encoder(F.normalize(directions, dim=-1))

        # Map positional encodings to latent features (MLP with skip connections)
        y = self._mlp(points_3d_encoded)
        y = self._fc1(y)

        # Calculate ray point density values
        densities_ray_points = self._sigma(y)
        densities_ray_points = densities_ray_points + torch.randn_like(densities_ray_points) * 0.1
        densities_ray_points = torch.relu(densities_ray_points)  # FIXME: Revise this

        # Calculate ray point rgb values
        y = torch.cat((y, directions_encoded[..., None, :].expand(-1, self._num_ray_points, -1)), dim=-1)
        y = self._fc2(y)
        rgbs_ray_points = self._rgb(y)

        # Rendering rgbs and densities along rays
        rgbs = self._renderer(rgbs_ray_points, densities_ray_points, points_3d)

        # Return pixel point rendered rgb
        return rgbs


class NerfModelRenderer:
    """Renders a novel synthesis view of a trained NeRF model for given camera."""

    def __init__(
        self, nerf_model: NerfModel, image_size: tuple[int, int], device: torch.device | None, dtype: torch.dtype | None
    ) -> None:
        """Constructor method.

        Args:
            nerf_model: NeRF model.
            image_size: image size.
            device: device to run the model on.
            dtype: dtype to run the model on.
        """
        self._nerf_model = nerf_model
        self._image_size = image_size
        self._device = device
        self._dtype = dtype

        self._pixels_grid, self._ones = self._create_pixels_grid()  # 1xHxWx2 and (H*W)x1

    def _create_pixels_grid(self) -> tuple[Tensor, Tensor]:
        """Creates the pixels grid to unproject to plane z=1.

        Args:
            image_size: image size: tuple[int, int]

        Returns:
            - Pixels grid: Tensor (1, H, W, 2)
            - Ones: Tensor (H*W, 1)
        """
        height, width = self._image_size
        pixels_grid: Tensor = create_meshgrid(
            height, width, normalized_coordinates=False, device=self._device, dtype=self._dtype
        )  # 1xHxWx2
        pixels_grid = pixels_grid.reshape(-1, 2)  # (H*W)x2

        ones = torch.ones(pixels_grid.shape[0], 1, device=pixels_grid.device, dtype=pixels_grid.dtype)  # (H*W)x1

        return pixels_grid, ones

    def render_view(self, camera: PinholeCamera) -> Tensor:
        """Renders a novel synthesis view of a trained NeRF model for given camera.

        Args:
            camera: camera for image rendering: PinholeCamera.

        Returns:
            Rendered image with shape :math:`(H, W, 3)`.
        """
        # create ray for this camera
        rays: Ray = self._create_rays(camera)

        # render the image
        with torch_inference_mode():
            rgb_model = self._nerf_model(rays.origin, rays.direction)

        rgb_image = rgb_model.view(self._image_size[0], self._image_size[1], 3)

        return rgb_image

    def _create_rays(self, camera: PinholeCamera) -> Ray:
        """Creates rays for a given camera.

        Args:
            camera: camera for image rendering: PinholeCamera.
        """
        height, width = self._image_size

        # convert to rays
        origin = camera.extrinsics[..., :3, -1]  # 1x3
        origin = origin.repeat(height * width, 1)  # (H*W)x3

        destination = camera.unproject(self._pixels_grid, self._ones)  # (H*W)x3

        return Ray.through(origin, destination)
