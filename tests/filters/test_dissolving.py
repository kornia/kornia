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

from kornia.core import Tensor
from kornia.filters.dissolving import StableDiffusionDissolving
from kornia.utils._compat import torch_version_le

WEIGHTS_CACHE_DIR = "weights/"


@pytest.mark.slow
@pytest.mark.skipif(
    torch_version_le(2, 0, 1),
    reason="Skipped for torch versions <= 2.0.1: transformers clip model needs distributed tensor.",
)
class TestStableDiffusionDissolving:
    @pytest.fixture(scope="class")
    def sdm_2_1(self):
        return StableDiffusionDissolving(version="2.1", cache_dir=WEIGHTS_CACHE_DIR)

    @pytest.fixture(scope="class")
    def dummy_image(self):
        # Create a dummy image tensor with shape [B, C, H, W], where B is the batch size.
        return torch.rand(1, 3, 64, 64)

    def test_init(self, sdm_2_1):
        assert isinstance(sdm_2_1, StableDiffusionDissolving), "Initialization failed"

    def test_encode_tensor_to_latent(self, sdm_2_1, dummy_image):
        latents = sdm_2_1.model.encode_tensor_to_latent(dummy_image)
        assert isinstance(latents, Tensor), "Latent encoding failed"
        assert latents.shape == (1, 4, 8, 8), "Latent shape mismatch"

    def test_decode_tensor_to_latent(self, sdm_2_1, dummy_image):
        latents = sdm_2_1.model.encode_tensor_to_latent(dummy_image)
        reconstructed_image = sdm_2_1.model.decode_tensor_to_latent(latents)
        assert isinstance(reconstructed_image, Tensor), "Latent decoding failed"
        assert reconstructed_image.shape == dummy_image.shape, "Reconstructed image shape mismatch"

    def test_dissolve(self, sdm_2_1, dummy_image):
        step_number = 500  # Test with a middle step
        dissolved_image = sdm_2_1(dummy_image, step_number)
        assert isinstance(dissolved_image, Tensor), "Dissolve failed"
        assert dissolved_image.shape == dummy_image.shape, "Dissolved image shape mismatch"

    def test_invalid_version(self):
        with pytest.raises(NotImplementedError):
            StableDiffusionDissolving(version="invalid_version")
