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

import pytest

from kornia.profiler.engines.augment import AugmentationPipeline


def test_duplicate_augmentation_names_raise_error() -> None:
    """Reject duplicate augmentation names in the configuration."""
    config = [
        {"name": "RandomRotation", "degrees": 15.0, "p": 1.0},
        {"name": "RandomRotation", "degrees": 30.0, "p": 1.0},
    ]

    with pytest.raises(ValueError, match="Augmentation names must be unique"):
        AugmentationPipeline(augmentations_config=config, mode="individual")
