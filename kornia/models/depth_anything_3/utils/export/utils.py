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

import numpy as np
import torch


def _denorm_and_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Denormalize to [0,255] and output (N, H, W, 3) uint8."""
    resnet_mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device
    )
    resnet_std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device
    )
    img = image_tensor * resnet_std[None, :, None, None] + resnet_mean[None, :, None, None]
    img = torch.clamp(img, 0.0, 1.0)
    img = (img.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)  # (N,H,W,3)
    return img
