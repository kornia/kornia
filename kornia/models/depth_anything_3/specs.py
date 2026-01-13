# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch


@dataclass
class Gaussians:
    """3DGS parameters, all in world space"""

    means: torch.Tensor  # world points, "batch gaussian dim"
    scales: torch.Tensor  # scales_std, "batch gaussian 3"
    rotations: torch.Tensor  # world_quat_wxyz, "batch gaussian 4"
    harmonics: torch.Tensor  # world SH, "batch gaussian 3 d_sh"
    opacities: torch.Tensor  # opacity | opacity SH, "batch gaussian" | "batch gaussian 1 d_sh"


@dataclass
class Prediction:
    depth: np.ndarray  # N, H, W
    is_metric: int
    sky: np.ndarray | None = None  # N, H, W
    conf: np.ndarray | None = None  # N, H, W
    extrinsics: np.ndarray | None = None  # N, 4, 4
    intrinsics: np.ndarray | None = None  # N, 3, 3
    processed_images: np.ndarray | None = None  # N, H, W, 3 - processed images for visualization
    gaussians: Gaussians | None = None  # 3D gaussians
    aux: dict[str, Any] = None  #
    scale_factor: Optional[float] = None  # metric scale
