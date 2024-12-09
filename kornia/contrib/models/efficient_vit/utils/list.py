# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

from typing import Any, Union


def val2list(x: Union[list[Any], tuple[Any, ...], Any], repeat_time: int = 1) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[list[Any], tuple[Any, ...], Any], min_len: int = 1, idx_repeat: int = -1) -> tuple[Any, ...]:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)
