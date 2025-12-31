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

"""Kornia Core — Core tensor operations, backends, and utilities for Kornia.

This subpackage provides core functionality, device management, and backend support.
"""

from . import external
from .mixin import (
    ImageModuleMixIn,
    ONNXExportMixin,
    ONNXMixin,
    ONNXRuntimeMixin,
)
from .module import ImageModule, ImageSequential
from .tensor_wrapper import TensorWrapper  # type: ignore

__all__ = [
    "ImageModule",
    "ImageModuleMixIn",
    "ImageSequential",
    "ONNXExportMixin",
    "ONNXMixin",
    "ONNXRuntimeMixin",
    "TensorWrapper",
    "external",
]
