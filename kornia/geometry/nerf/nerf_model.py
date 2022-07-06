import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_dims, n_units: int = 2, n_unit_layers: int = 4, n_hidden: int = 128):
        super().__init__()
        self._n_unit_layers = n_unit_layers
        layers = []
        for i in range(n_units):
            n_unit_inp_dims = n_dims if i == 0 else n_hidden + n_dims
            for j in range(n_unit_layers):
                n_layer_inp_dims = n_unit_inp_dims if j == 0 else n_hidden
                layer = nn.Linear(n_layer_inp_dims, n_hidden)
                nn.init.xavier_uniform_(layer.weight.data)  # FIXME: Verify proper Xavier weight initialization!
                layers.append(nn.Sequential(layer, nn.ReLU(True)))
        self._mlp = nn.ModuleList(layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp
        inp_skip = inp
        for i, layer in enumerate(self._mlp):
            if i > 0 and i % self._n_unit_layers == 0:
                out = torch.cat((out, inp_skip), dim=-1)
            out = layer(out)
        return out


class NerfModel(nn.Module):
    def __init__(self, n_pos_dims, n_dir_dims):
        super().__init__()

    def forward(self):
        pass
        # Sample xyz for ray parameters

        # Encode positions

        # Map positional encodings to latent features (MLP with skip connections)

        # Return sample point color and density
