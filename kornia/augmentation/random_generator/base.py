from typing import Dict

import torch
import torch.nn as nn


class _PostInitInjectionMetaClass(type):
    """To inject the ``__post_init__`` function after the creation of each instance."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class RandomGeneratorBase(nn.Module, metaclass=_PostInitInjectionMetaClass):
    """Base class for generating random augmentation parameters."""

    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self) -> None:
        self.set_rng_device_and_dtype()

    def set_rng_device_and_dtype(
        self, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32
    ) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.make_samplers(device, dtype)

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        raise NotImplementedError

    def forward(self, batch_shape: torch.Size, same_on_batch: bool = False) -> Dict[str, torch.Tensor]:  # type:ignore
        raise NotImplementedError
