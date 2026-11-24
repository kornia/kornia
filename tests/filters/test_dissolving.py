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

from kornia.filters.dissolving import StableDiffusionDissolving, _DissolvingWraper_HF

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


class TestStableDiffusionDissolvingPrecision(BaseTester):
    @pytest.mark.parametrize(
        "version, model_dtype, force_upcast, expected_vae_dtype",
        [
            pytest.param("xl", torch.float16, True, torch.float32, id="sdxl-upcast"),
            pytest.param("xl", torch.float16, False, torch.float16, id="sdxl-half-safe"),
            pytest.param("xl", torch.float32, True, torch.float32, id="sdxl-float32"),
            pytest.param("xl", torch.bfloat16, True, torch.bfloat16, id="sdxl-bfloat16"),
            pytest.param("1.5", torch.float16, True, torch.float16, id="sd15-half"),
        ],
    )
    def test_vae_unet_precision_boundaries_6826(
        self, version, model_dtype, force_upcast, expected_vae_dtype, device, dtype, monkeypatch
    ):
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(force_upcast=force_upcast, scaling_factor=0.18215)
                self.encoder = torch.nn.Conv2d(3, 3, 1, bias=False, device=device, dtype=model_dtype)
                self.post_quant_conv = torch.nn.Conv2d(3, 3, 1, bias=False, device=device, dtype=model_dtype)
                self.decoder = torch.nn.Conv2d(3, 3, 1, bias=False, device=device, dtype=model_dtype)
                for layer in (self.encoder, self.post_quant_conv, self.decoder):
                    with torch.no_grad():
                        layer.weight.copy_(torch.eye(3, device=device, dtype=model_dtype).view(3, 3, 1, 1))

            @property
            def dtype(self):
                return self.encoder.weight.dtype

            def encode(self, image):
                encoded = self.encoder(image)
                return {"latent_dist": SimpleNamespace(sample=lambda: encoded)}

            def decode(self, latents):
                return {"sample": self.decoder(self.post_quant_conv(latents))}

        # Real small convolutions make a mixed input/weight dtype fail at the same
        # boundary as the checkpoint, without downloading it or emulating its weights.
        denoiser = torch.nn.Conv2d(3, 3, 1, device=device, dtype=model_dtype)
        unet = Mock(side_effect=lambda latent, *args, **kwargs: SimpleNamespace(sample=denoiser(latent)))
        unet.dtype = model_dtype
        scheduler = Mock(timesteps=[999], alphas_cumprod=torch.ones(1001, device=device))
        scheduler.scale_model_input.side_effect = lambda latent, timestep: latent
        pipeline = SimpleNamespace(vae=VAE(), unet=unet, tokenizer=Mock(), scheduler=scheduler, device=device)
        pipeline_class = Mock()
        pipeline_class.from_pretrained.return_value = pipeline
        monkeypatch.setattr(
            "kornia.filters.dissolving.diffusers",
            SimpleNamespace(
                DDIMScheduler=Mock(), StableDiffusionPipeline=pipeline_class, StableDiffusionXLPipeline=pipeline_class
            ),
        )
        model = StableDiffusionDissolving(version=version)
        assert {parameter.dtype for parameter in pipeline.vae.parameters()} == {expected_vae_dtype}
        assert denoiser.weight.dtype == model_dtype
        wrapper = model.model
        wrapper.init_prompt = Mock()
        wrapper.context = torch.zeros(2, 1, 4, device=device, dtype=model_dtype)
        wrapper.pooled_embeddings = torch.zeros(2, 4, device=device, dtype=model_dtype)
        wrapper.add_time_ids = torch.zeros(2, 6, device=device, dtype=model_dtype)
        image = torch.tensor([0.0, 0.25, 0.75, 1.0], device=device, dtype=dtype).view(1, 1, 2, 2)
        image = image.expand(2, 3, 2, 2).clone().requires_grad_()

        output = wrapper.dissolve(image, 0)

        assert unet.call_args.args[0].dtype == model_dtype
        assert output.dtype == expected_vae_dtype
        assert output.device == image.device
        assert not output.requires_grad
        assert torch.isfinite(output).all()
        # Alpha=1 leaves the encoded image unchanged; only VAE dtype roundoff remains.
        self.assert_close(output, image.to(expected_vae_dtype), atol=torch.finfo(expected_vae_dtype).eps, rtol=0)


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


class TestStableDiffusionDissolvingLatentScale(BaseTester):
    @pytest.mark.parametrize(
        "scaling_factor, expected_scaled",
        [
            pytest.param(0.18215, [-0.3643, 0.0, 0.18215, 0.7286], id="sd15"),
            pytest.param(0.13025, [-0.2605, 0.0, 0.13025, 0.521], id="sdxl"),
            pytest.param(0.5, [-1.0, 0.0, 0.5, 2.0], id="custom-vae"),
        ],
    )
    def test_convention_latents_use_vae_scaling_factor(self, scaling_factor, expected_scaled, device, dtype):
        # The SDXL VAE config specifies 0.13025; SD 1.5 uses 0.18215.
        # Pin both directions separately: an encode/decode round trip alone hides
        # a wrong scale when the same constant is used in both directions.
        latent = torch.tensor([-2.0, 0.0, 1.0, 4.0], device=device, dtype=dtype).view(1, 1, 2, 2).repeat(2, 4, 1, 1)
        image = torch.full((2, 3, 16, 16), 0.25, device=device, dtype=dtype)
        decoded = torch.full_like(image, -0.5)
        vae = SimpleNamespace(
            dtype=dtype,
            config=SimpleNamespace(scaling_factor=scaling_factor),
            encode=Mock(return_value={"latent_dist": SimpleNamespace(sample=lambda: latent)}),
            decode=Mock(return_value={"sample": decoded}),
        )
        model = SimpleNamespace(
            device=device,
            vae=vae,
            tokenizer=None,
            scheduler=SimpleNamespace(set_timesteps=Mock(), timesteps=[0]),
        )
        wrapper = _DissolvingWraper_HF(model)
        scaled = torch.tensor(expected_scaled, device=device, dtype=dtype).view(1, 1, 2, 2).repeat(2, 4, 1, 1)
        original = scaled.clone()

        encoded = wrapper.encode_tensor_to_latent(image)
        self.assert_close(encoded, scaled)
        self.assert_close(vae.encode.call_args.args[0], image.new_full(image.shape, -0.5))

        output = wrapper.decode_tensor_to_latent(scaled)
        self.assert_close(vae.decode.call_args.args[0], latent)
        self.assert_close(output, image)
        # Decoding must not rescale the caller's latent tensor in place.
        self.assert_close(scaled, original)
