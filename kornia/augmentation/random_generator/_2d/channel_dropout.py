from __future__ import annotations

import torch

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check, _range_bound
from kornia.core import Tensor


class ChannelDropoutGenerator(RandomGeneratorBase):
    r"""Generate random dropout masks for channels in a batch of images.

    Args:
        num_drop_channels: The number of channels to drop randomly.

    Returns:
        A dictionary containing the dropout mask.
            - dropout_mask: Binary masks (bool) indicating the dropped channels with a shape of (B, C, H, W).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.
    """

    def __init__(
        self,
        num_drop_channels: int,
    ) -> None:
        super().__init__()
        self.num_drop_channels = num_drop_channels
        self.drop_sampler: UniformDistribution

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        repr_buf = f"num_drop_channels={self.num_drop_channels}"
        return repr_buf

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        r"""Create samplers for generating random dropout parameters."""
        drop = _range_bound((0.0, 1.0), "drop", device=device, dtype=dtype)
        self.drop_sampler = UniformDistribution(drop[0], drop[1], validate_args=False)

    def forward(self, batch_shape: tuple[int, ...], same_on_batch: bool = False) -> dict[str, Tensor]:
        r"""Generates a mask for dropout channels."""
        batch_size, channels, _, _ = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = self.device, self.dtype

        batch_idx = torch.arange(batch_size, device=_device, dtype=torch.long).reshape(batch_size, 1)
        channel_idx = torch.argsort(
            _adapted_rsampling((batch_size, channels), self.drop_sampler, same_on_batch), dim=1
        )[:, : self.num_drop_channels].to(torch.long)

        return {
            "batch_idx": batch_idx,
            "channel_idx": channel_idx,
        }
