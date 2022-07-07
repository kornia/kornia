import torch
from torch import nn

from kornia.geometry.nerf.positional_encoder import PositionalEncoder
from kornia.geometry.nerf.rays import sample_lengths, sample_ray_points


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
                layers.append(nn.Sequential(layer, nn.ReLU(True)))
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
        num_pos_freqs: int,  # FIXME: Decide on defaults for arguments
        num_dir_freqs: int,
        num_units: int,
        num_nuit_layers: int,
        num_hidden: int,
    ):
        super().__init__()
        self._num_ray_points = num_ray_points
        self._pos_encoder = PositionalEncoder(num_pos_freqs)
        self._dir_encoder = PositionalEncoder(num_dir_freqs)
        self._mlp = MLP(self._pos_encoder.num_encoded_dims, num_units, num_nuit_layers, num_hidden)

    def forward(self, origins: torch.Tensor, directions: torch.Tensor):

        # Sample xyz for ray parameters
        batch_size = origins.shape[0]
        lengths = sample_lengths(
            batch_size, self._num_ray_points, irregular=False
        )  # FIXME: handle the case of irregular smapling along rays, and hierarchical sampling
        points_3d = sample_ray_points(
            origins, directions, lengths
        )  # FIXME: Normalize points to [-1, 1] before encoding?

        # Encode positions & directions
        points_3d_encoded = self._pos_encoder(points_3d)
        directions_encoded = self._dir_encoder(directions)  # FIXME: Normalize directions to [-1, 1] before encoding?

        # Map positional encodings to latent features (MLP with skip connections)
        y = self._mlp(points_3d_encoded)

        print(directions_encoded, y)  # FIXME: Remove this line

        # Return sample point color and density
