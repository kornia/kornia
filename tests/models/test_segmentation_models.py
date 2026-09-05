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
import torch

SMP_REASON = '`segmentation_models_pytorch` is not installed. Install it with: pip install "kornia[segmentation]"'


class TestSegmentationModelsBuilder:
    """Smoke tests for the ``kornia[segmentation]`` extra.

    They skip unless ``segmentation_models_pytorch`` is installed; they exist so the builder cannot
    rot silently once somebody installs the extra.
    """

    def test_build_unet(self, device, dtype):
        pytest.importorskip("segmentation_models_pytorch", reason=SMP_REASON)

        from kornia.models.segmentation import SegmentationModelsBuilder, SemanticSegmentation

        model = SegmentationModelsBuilder.build(
            model_name="Unet", encoder_name="resnet34", classes=1, encoder_weights=None
        )
        assert isinstance(model, SemanticSegmentation)

        model = model.to(device, dtype)
        images = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        out = model(images)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 1, 64, 64)
