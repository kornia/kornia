# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

from functools import partial
from typing import Any, Optional, Union

from torch import nn

from kornia.contrib.models.efficient_vit.utils import build_kwargs_from_config

# register activation function here
REGISTERED_ACT_DICT: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),  # type: ignore
}


def build_act(name: Optional[str], **kwargs: dict[str, Any]) -> Union[nn.Module, None]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)

    return None
