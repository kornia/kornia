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

import os
from typing import Any

import torch
from torch import nn

from kornia.core import ImageModule
from kornia.core.external import diffusers


class _DissolvingWraper_HF:
    def __init__(self, model: nn.Module, num_ddim_steps: int = 50, is_sdxl: bool = False) -> None:
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.is_sdxl = is_sdxl
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.total_steps = len(self.model.scheduler.timesteps)  # Total number of sampling steps.
        self.prompt: str
        self.context: torch.Tensor
        self.pooled_embeddings: torch.Tensor | None = None
        self.add_time_ids: torch.Tensor | None = None

    def predict_start_from_noise(self, noise_pred: torch.Tensor, timestep: int, latent: torch.Tensor) -> torch.Tensor:
        return (
            torch.sqrt(1.0 / self.model.scheduler.alphas_cumprod[timestep]) * latent
            - torch.sqrt(1.0 / self.model.scheduler.alphas_cumprod[timestep] - 1) * noise_pred
        )

    @torch.no_grad()
    def init_prompt(self, prompt: str) -> None:
        if self.is_sdxl:
            # Handle models with dual text encoders
            tokenizers = (
                [self.model.tokenizer, self.model.tokenizer_2]
                if hasattr(self.model, "tokenizer_2")
                else [self.model.tokenizer]
            )
            text_encoders = (
                [self.model.text_encoder, self.model.text_encoder_2]
                if hasattr(self.model, "text_encoder_2")
                else [self.model.text_encoder]
            )

            prompt_embeds_list = []
            pooled_prompt_embeds = None

            for i, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
                text_inputs = tokenizer(
                    [prompt],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(self.model.device)

                outputs = text_encoder(text_input_ids, output_hidden_states=True)
                # Extract pooled embeddings from the last text encoder
                if i == len(tokenizers) - 1:
                    # Get pooled output from text encoder
                    pooled_prompt_embeds = (
                        outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.text_embeds
                    )
                # Get the penultimate hidden state
                prompt_embeds = outputs.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            # Get negative (unconditional) embeddings
            negative_prompt_embeds_list = []
            negative_pooled_prompt_embeds = None

            for i, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
                uncond_inputs = tokenizer(
                    [""],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_inputs.input_ids.to(self.model.device)

                outputs = text_encoder(uncond_input_ids, output_hidden_states=True)
                # Extract pooled embeddings from the last text encoder
                if i == len(tokenizers) - 1:
                    negative_pooled_prompt_embeds = (
                        outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.text_embeds
                    )
                negative_prompt_embeds = outputs.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

            self.context = torch.cat([negative_prompt_embeds, prompt_embeds])
            # Type narrowing: these are guaranteed to be set in the loops above when is_sdxl=True
            if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
                self.pooled_embeddings = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

            # Create time_ids for models that require additional conditioning
            # Format: (original_size, crops_coords_top_left, target_size)
            add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=self.model.device)
            self.add_time_ids = torch.cat([add_time_ids, add_time_ids])
        else:
            # SD 1.x
            uncond_input = self.model.tokenizer(
                [""], padding="max_length", max_length=self.model.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
            text_input = self.model.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
            self.context = torch.cat([uncond_embeddings, text_embeddings])

        self.prompt = prompt

    # Encode the image to latent using the VAE.
    @torch.no_grad()
    def encode_tensor_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image = (image / 0.5 - 1).to(self.model.device)
            latents = self.model.vae.encode(image)["latent_dist"].sample()
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def decode_tensor_to_latent(self, latents: torch.Tensor) -> torch.Tensor:
        # Perform in-place detach to reduce memory usage and copies
        latents = latents.detach()
        latents = latents * (1.0 / 0.18215)  # Fused division as multiplication (faster)
        # Reduce attribute lookups by localizing frequently used attributes
        vae_decode = self.model.vae.decode
        image = vae_decode(latents)["sample"]
        # Use in-place arithmetic/clamp for throughput
        image = image.div_(2).add_(0.5)
        image.clamp_(0, 1)
        return image

    @torch.no_grad()
    def one_step_dissolve(self, latent: torch.Tensor, i: int) -> torch.Tensor:
        _, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        # NOTE: This implementation use a reversed timesteps but can reach to
        # a stable dissolving effect.
        t = self.num_ddim_steps - self.model.scheduler.timesteps[i]
        latent = self.model.scheduler.scale_model_input(latent, t)
        cond_embeddings = cond_embeddings.repeat(latent.size(0), 1, 1)

        if self.is_sdxl:
            # Pass additional conditioning to models that require it
            # Type narrowing: these are guaranteed to be set when is_sdxl=True
            if self.pooled_embeddings is not None and self.add_time_ids is not None:
                _, pooled_embeds = self.pooled_embeddings.chunk(2)
                _, add_time_ids = self.add_time_ids.chunk(2)

                # Expand embeddings to match batch size if needed
                batch_size = latent.size(0)
                if pooled_embeds.size(0) != batch_size:
                    pooled_embeds = pooled_embeds.expand(batch_size, -1)
                if add_time_ids.size(0) != batch_size:
                    add_time_ids = add_time_ids.expand(batch_size, -1)

                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids,
                }
                noise_pred = self.model.unet(latent, t, cond_embeddings, added_cond_kwargs=added_cond_kwargs).sample
        else:
            noise_pred = self.model.unet(latent, t, cond_embeddings).sample

        pred_x0 = self.predict_start_from_noise(noise_pred, t, latent)
        return pred_x0

    @torch.no_grad()
    def dissolve(self, image: torch.Tensor, t: int) -> torch.Tensor:
        self.init_prompt("")
        latent = self.encode_tensor_to_latent(image)
        ddim_latents = self.one_step_dissolve(latent, t)
        dissolved = self.decode_tensor_to_latent(ddim_latents)
        return dissolved


class StableDiffusionDissolving(ImageModule):
    r"""Perform dissolving transformation using StableDiffusion models.

    Based on :cite:`shi2024dissolving`, the dissolving transformation is essentially applying one-step
    reverse diffusion. Our implementation currently supports HuggingFace implementations of SD 1.4, 1.5
    and SD XL (replacing the discontinued SD 2.1). SD 1.X tends to remove more details than SD-XL.

    .. list-table:: Title
        :widths: 32 32 32
        :header-rows: 1

        * - SD 1.4
          - SD 1.5
          - SD XL
        * - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.4.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.5.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-2.1.png

    Args:
        version: the version of the stable diffusion model. Options: "1.4", "1.5", "xl".
        **kwargs: additional arguments for `.from_pretrained`.

    """

    def __init__(self, version: str = "1.5", **kwargs: Any):
        super().__init__()
        DDIMScheduler = diffusers.DDIMScheduler

        # Load the scheduler and model pipeline from diffusers library
        scheduler = DDIMScheduler(  # type:ignore
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Filter out arguments that are not supported by all component models
        kwargs.pop("offload_state_dict", None)

        # Get HF token from environment if not explicitly provided
        if "token" not in kwargs:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                kwargs["token"] = hf_token

        is_sdxl = False

        if version == "1.4":
            StableDiffusionPipeline = diffusers.StableDiffusionPipeline
            self._sdm_model = StableDiffusionPipeline.from_pretrained(  # type:ignore
                "CompVis/stable-diffusion-v1-4", scheduler=scheduler, **kwargs
            )
        elif version == "1.5":
            StableDiffusionPipeline = diffusers.StableDiffusionPipeline
            self._sdm_model = StableDiffusionPipeline.from_pretrained(  # type:ignore
                "runwayml/stable-diffusion-v1-5", scheduler=scheduler, **kwargs
            )
        elif version == "xl":
            StableDiffusionXLPipeline = diffusers.StableDiffusionXLPipeline
            self._sdm_model = StableDiffusionXLPipeline.from_pretrained(  # type:ignore
                "stabilityai/stable-diffusion-xl-base-1.0", scheduler=scheduler, **kwargs
            )
            is_sdxl = True
        else:
            raise NotImplementedError

        self.model = _DissolvingWraper_HF(self._sdm_model, num_ddim_steps=1000, is_sdxl=is_sdxl)

    def forward(self, input: torch.Tensor, step_number: int) -> torch.Tensor:
        return self.model.dissolve(input, step_number)
