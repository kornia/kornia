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

"""Models submodule for Kornia.

This package provides model architectures and utilities for tasks such as depth estimation,
detection, segmentation, super-resolution, and tracking.
"""

from . import (
    depth_estimation,
    detection,
    segmentation,
    super_resolution,
    tracking,
)
from .dexined import DexiNed
from .processors import *
from .structures import Prompts, SegmentationResults
from .tiny_vit import TinyViT
from .vit import VisionTransformer
from .vit_mobile import MobileViT
from .yunet import YuNet
