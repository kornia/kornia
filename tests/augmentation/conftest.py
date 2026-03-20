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


@pytest.fixture(autouse=True)
def xfail_bfloat16_augmentation(request):
    """Mark augmentation tests as xfail when bfloat16 is used.

    AugmentationBase2D.validate_tensor only accepts float16/float32/float64.
    """
    if "dtype" not in request.fixturenames:
        return
    dtype = request.getfixturevalue("dtype")
    if dtype == torch.bfloat16:
        request.applymarker(
            pytest.mark.xfail(reason="AugmentationBase2D does not support bfloat16", strict=False)
        )
