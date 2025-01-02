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

from __future__ import annotations

from typing import Any, TypeVar

from kornia.core import Device, Dtype, Tensor

T = TypeVar("T")


def dict_to(data: dict[T, Any], device: Device, dtype: Dtype) -> dict[T, Any]:
    out: dict[T, Any] = {}
    for key, val in data.items():
        out[key] = val.to(device, dtype) if isinstance(val, Tensor) else val
    return out
