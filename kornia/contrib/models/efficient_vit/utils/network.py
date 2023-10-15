# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

from inspect import signature
from typing import Any, Union


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    if isinstance(kernel_size, (tuple,)):
        return tuple([get_same_padding(ks) for ks in kernel_size])  # type: ignore

    # assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_kwargs_from_config(config: dict[str, Any], target_func: Any) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs
