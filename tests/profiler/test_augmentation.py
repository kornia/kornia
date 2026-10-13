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

import torch
from torch import Tensor, nn

from kornia.profiler.representation_profiler import model_profile_under_augmentation


class SimpleModel(nn.Module):
    """Simple convolutional model used to test augmentation profiling."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, activation, pooling, and flattening."""
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.flatten(1)


def test_model_profile_under_augmentation() -> None:
    """Test profiling of individual augmentations and result columns."""
    torch.manual_seed(42)

    model = SimpleModel()
    input_tensor = torch.rand(2, 3, 32, 32)

    config = {
        "input": input_tensor,
        "augmentations": [
            {
                "name": "rotation",
                "degrees": 15,
            },
            {
                "name": "brightness",
                "brightness_factor": (0.8, 1.2),
            },
        ],
        "mode": "individual",
        "layers": ["conv", "pool"],
        "processing": "flatten",
        "metrics": ["cosine", "linear"],
    }

    profiler = model_profile_under_augmentation(model, config=config)
    results = profiler.df

    assert results is not None
    assert results.shape == (2, 5)
    assert set(results["tag"]) == {"rotation", "brightness"}

    expected_columns = {
        "tag",
        "cosine_conv",
        "linear_conv",
        "cosine_pool",
        "linear_pool",
    }
    assert set(results.columns) == expected_columns
