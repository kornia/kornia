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

"""EfficientLoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed.

This module is adapted from the original EfficientLoFTR implementation at commit ffd4a4644 (2025-07-30), its last
release under the Apache License 2.0:
    https://github.com/zju3dv/EfficientLoFTR/tree/ffd4a4644064354468eb1f0c7a3e732233cb732f

Original work by the EfficientLoFTR authors, licensed under the Apache License 2.0.
Modifications copyright (c) 2025 Kornia Team, licensed under the Apache License 2.0.

Reference:
    Yifan Wang, Xingyi He, Sida Peng, Dongli Tan, Xiaowei Zhou.
    "Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed", CVPR 2024.
"""

from __future__ import annotations

from .efficient_loftr import EfficientLoFTR
