import torch
from torch import nn
from torch.nn import functional as F

from kornia.geometry.nerf.positional_encoder import PositionalEncoder
from kornia.geometry.nerf.rays import sample_lengths, sample_ray_points
from kornia.geometry.nerf.renderer import IrregularRenderer, RegularRenderer


class MLP(nn.Module):
    def __init__(self, num_dims, num_units: int = 2, num_unit_layers: int = 4, num_hidden: int = 128):
        super().__init__()
        self._num_unit_layers = num_unit_layers
        layers = []
        for i in range(num_units):
            n_unit_inp_dims = num_dims if i == 0 else num_hidden + num_dims
            for j in range(num_unit_layers):
                num_layer_inp_dims = n_unit_inp_dims if j == 0 else num_hidden
                layer = nn.Linear(num_layer_inp_dims, num_hidden)
                # nn.init.xavier_uniform_(layer.weight.data)  # FIXME: Verify proper Xavier weight initialization!
                layers.append(nn.Sequential(layer, nn.ReLU()))
        self._mlp = nn.ModuleList(layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp
        inp_skip = inp
        for i, layer in enumerate(self._mlp):
            if i > 0 and i % self._num_unit_layers == 0:
                out = torch.cat((out, inp_skip), dim=-1)
            out = layer(out)
        return out


class NerfModel(nn.Module):
    def __init__(
        self,
        num_ray_points: int,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        num_units: int = 2,
        num_unit_layers: int = 4,
        num_hidden: int = 256,
    ):
        super().__init__()
        self._num_ray_points = num_ray_points
        self._irregular_ray_sampling = False
        self._renderer = IrregularRenderer() if self._irregular_ray_sampling else RegularRenderer()

        self._pos_encoder = PositionalEncoder(3, num_pos_freqs)
        self._dir_encoder = PositionalEncoder(3, num_dir_freqs)
        self._mlp = MLP(self._pos_encoder.num_encoded_dims, num_units, num_unit_layers, num_hidden)
        self._fc1 = nn.Linear(num_hidden, num_hidden)
        self._fc2 = nn.Sequential(
            nn.Linear(num_hidden + self._dir_encoder.num_encoded_dims, num_hidden // 2), nn.ReLU()
        )

        sigma = nn.Linear(num_hidden, 1, bias=True)
        torch.nn.init.xavier_uniform_(sigma.weight.data)

        sigma.bias.data = torch.tensor([0.0]).float()

        # self._sigma = nn.Sequential(sigma, nn.ReLU())  # FIXME: Revise this
        self._sigma = sigma

        self._rgb = nn.Sequential(nn.Linear(num_hidden // 2, 3), nn.Sigmoid())

        # self._debug = nn.Linear(3, 3)  # FIXME: Remove this line

    def forward(self, origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:

        # Sample xyz for ray parameters
        batch_size = origins.shape[0]
        lengths = sample_lengths(
            batch_size, self._num_ray_points, device=origins.device, irregular=self._irregular_ray_sampling
        )  # FIXME: handle the case of hierarchical sampling
        points_3d = sample_ray_points(origins, directions, lengths)

        # return self._debug(points_3d)   # FIXME: Remove this line

        # Encode positions & directions
        points_3d_encoded = self._pos_encoder(points_3d)
        directions_encoded = self._dir_encoder(
            F.normalize(directions, dim=-1)
        )  # FIXME: Normalize directions before encoding?

        # Map positional encodings to latent features (MLP with skip connections)
        y = self._mlp(points_3d_encoded)
        y = self._fc1(y)
        densities_ray_points = self._sigma(y)
        # densities_ray_points = densities_ray_points + torch.randn_like(densities_ray_points) * 0.1
        # densities_ray_points = torch.relu(densities_ray_points)  # FIXME: Revise this

        y = torch.cat((y, directions_encoded[..., None, :].expand(-1, self._num_ray_points, -1)), dim=-1)
        y = self._fc2(y)
        rgbs_ray_points = self._rgb(y)

        # Rendring rgbs and densities along rays
        rgbs = self._renderer(rgbs_ray_points, densities_ray_points, points_3d)

        # Return sample point color and density
        return rgbs
