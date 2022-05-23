import torch
from torch import nn


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self._d_input = d_input
        self._n_freqs = n_freqs
        self._log_space = log_space
        self._d_output = d_input * (1 + 2 * self._n_freqs)
        self._embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self._log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self._n_freqs - 1, self._n_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0 ** (self._n_freqs - 1), self._n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self._embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self._embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        if x.ndim < 2 or x.shape[1] != self._d_input:
            raise RuntimeError('Invalid input geometry to encode')
        return torch.concat([fn(x) for fn in self._embed_fns], dim=-1)
