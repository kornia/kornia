# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

import os
from inspect import signature

import torch
from torch import nn

__all__ = ["is_parallel", "get_device", "get_same_padding", "build_kwargs_from_config", "load_state_dict_from_file"]


def is_parallel(model: nn.Module) -> bool:
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])

    # assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint
