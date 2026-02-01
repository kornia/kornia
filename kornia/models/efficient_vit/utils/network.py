# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
from __future__ import annotations

from inspect import signature
from typing import Any, Union


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    """Return padding values."""
    if isinstance(kernel_size, (tuple,)):
        return tuple([get_same_padding(ks) for ks in kernel_size])  # type: ignore

    # assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_kwargs_from_config(config: dict[str, Any], target_func: Any) -> dict[str, Any]:
    """Return kwargs from config object."""
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key, value in config.items():
        if key in valid_keys:
            kwargs[key] = value
    return kwargs
