import torch
from torch import nn
from torch.nn import functional as F

from kornia.geometry.nerf.positional_encoder import PositionalEncoder
from kornia.geometry.nerf.rays import calc_ray_t_vals, sample_lengths, sample_ray_points
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
                nn.init.xavier_uniform_(layer.weight.data)  # FIXME: Verify proper Xavier weight initialization!
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
        self._fc1 = nn.Linear(num_hidden, num_hidden)  # FIXME: Relu activation for FC1?
        self._fc2 = nn.Sequential(
            nn.Linear(num_hidden + self._dir_encoder.num_encoded_dims, num_hidden // 2), nn.ReLU()
        )
        self._sigma = nn.Linear(num_hidden, 1)
        self._rgb = nn.Linear(num_hidden // 2, 3)

    def forward(self, origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:

        # Sample xyz for ray parameters
        batch_size = origins.shape[0]
        lengths = sample_lengths(
            batch_size, self._num_ray_points, irregular=self._irregular_ray_sampling
        )  # FIXME: handle the case of hierarchical sampling
        points_3d = sample_ray_points(origins, directions, lengths)

        # Encode positions & directions
        points_3d_encoded = self._pos_encoder(points_3d)
        directions_encoded = self._dir_encoder(F.normalize(directions, dim=-1))

        # Map positional encodings to latent features (MLP with skip connections)
        y = self._mlp(points_3d_encoded)
        y = self._fc1(y)
        densities_ray_points = self._sigma(y)

        y = torch.cat((y, directions_encoded[..., None, :].expand(-1, self._num_ray_points, -1)), dim=-1)
        y = self._fc2(y)
        rgbs_ray_points = self._rgb(y)

        # Rendring rgbs and densities along rays
        if (
            self._irregular_ray_sampling
        ):  # FIXME: Consolidate irregular and regular renderer to one unique forward API that will get point_3d as an
            # additional input, and will internally calculatite either deltas to t_vals
            t_vals = calc_ray_t_vals(points_3d)
            rgbs = self._renderer(rgbs_ray_points, densities_ray_points, t_vals)
        else:
            deltas = torch.ones(batch_size) * 1.0 / (self._num_ray_points - 1)
            rgbs = self._renderer(rgbs_ray_points, densities_ray_points, deltas)

        # Return sample point color and density
        return rgbs
