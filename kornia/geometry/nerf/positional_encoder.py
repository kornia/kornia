import torch
from torch import nn


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, num_freqs: int, log_space: bool = False):
        super().__init__()
        self._embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0 ** (num_freqs - 1), num_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self._embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))  # FIXME: PI?
            self._embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        # FIXME: Make sure x in normalized [-1, 1]!!
        return torch.concat([fn(x) for fn in self._embed_fns], dim=-1)
