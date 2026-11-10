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

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kornia.filters.dissolving import StableDiffusionDissolving

from testing.base import BaseTester

WEIGHTS_CACHE_DIR = "weights/"


@pytest.mark.device_agnostic
class TestStableDiffusionDissolvingVersion(BaseTester):
    @pytest.fixture
    def diffusers_stub(self, monkeypatch):
        stub = SimpleNamespace(DDIMScheduler=Mock(), StableDiffusionPipeline=Mock(), StableDiffusionXLPipeline=Mock())
        for pipeline in (stub.StableDiffusionPipeline, stub.StableDiffusionXLPipeline):
            pipeline.from_pretrained.return_value.scheduler.timesteps = [0]
        monkeypatch.setattr("kornia.filters.dissolving.diffusers", stub)
        return stub

    @pytest.mark.parametrize("version", ["invalid_version", "2.1", ""])
    def test_exception(self, version, diffusers_stub):
        message = f"Unsupported Stable Diffusion version {version!r}. Supported versions are '1.4', '1.5', and 'xl'."
        with pytest.raises(NotImplementedError, match=re.escape(message)):
            StableDiffusionDissolving(version=version)
        diffusers_stub.StableDiffusionPipeline.from_pretrained.assert_not_called()
        diffusers_stub.StableDiffusionXLPipeline.from_pretrained.assert_not_called()

    @pytest.mark.parametrize(
        "version, model_id",
        [
            ("1.4", "CompVis/stable-diffusion-v1-4"),
            ("1.5", "runwayml/stable-diffusion-v1-5"),
            ("xl", "stabilityai/stable-diffusion-xl-base-1.0"),
        ],
    )
    def test_supported_version(self, version, model_id, diffusers_stub):
        model = StableDiffusionDissolving(version=version)
        pipeline = (
            diffusers_stub.StableDiffusionXLPipeline if version == "xl" else diffusers_stub.StableDiffusionPipeline
        )
        pipeline.from_pretrained.assert_called_once()
        assert pipeline.from_pretrained.call_args.args == (model_id,)
        assert model.model.is_sdxl is (version == "xl")


@pytest.mark.slow
class TestStableDiffusionDissolving(BaseTester):
    @pytest.fixture(scope="class")
    def sdm_2_1(self):
        pytest.importorskip(
            "diffusers", reason='`diffusers` is not installed. Install it with: pip install "kornia[sd]"'
        )
        return StableDiffusionDissolving(version="1.5", cache_dir=WEIGHTS_CACHE_DIR)

    @pytest.fixture(scope="class")
    def dummy_image(self):
        # Create a dummy image tensor with shape [B, C, H, W], where B is the batch size.
        return torch.rand(1, 3, 64, 64)

    def test_init(self, sdm_2_1):
        assert isinstance(sdm_2_1, StableDiffusionDissolving), "Initialization failed"

    def test_encode_tensor_to_latent(self, sdm_2_1, dummy_image, device, dtype):
        dummy_image = dummy_image.to(device, dtype)
        # Note: the model internally converts latents to its own dtype, so we don't strictly check output dtype match
        latents = sdm_2_1.model.encode_tensor_to_latent(dummy_image)
        assert isinstance(latents, torch.Tensor), "Latent encoding failed"
        assert latents.shape == (1, 4, 8, 8), "Latent shape mismatch"

    def test_decode_tensor_to_latent(self, sdm_2_1, dummy_image, device, dtype):
        dummy_image = dummy_image.to(device, dtype)
        latents = sdm_2_1.model.encode_tensor_to_latent(dummy_image)
        reconstructed_image = sdm_2_1.model.decode_tensor_to_latent(latents)
        assert isinstance(reconstructed_image, torch.Tensor), "Latent decoding failed"
        assert reconstructed_image.shape == dummy_image.shape, "Reconstructed image shape mismatch"

    def test_dissolve(self, sdm_2_1, dummy_image, device, dtype):
        dummy_image = dummy_image.to(device, dtype)
        step_number = 500  # Test with a middle step
        dissolved_image = sdm_2_1(dummy_image, step_number)
        assert isinstance(dissolved_image, torch.Tensor), "Dissolve failed"
        assert dissolved_image.shape == dummy_image.shape, "Dissolved image shape mismatch"
