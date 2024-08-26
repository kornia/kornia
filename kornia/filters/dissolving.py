from typing import Any

import torch

from kornia.core import ImageModule, Module, Tensor
from kornia.core.external import diffusers


class _DissolvingWraper_HF:
    def __init__(self, model: Module, num_ddim_steps: int = 50) -> None:
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.total_steps = len(self.model.scheduler.timesteps)  # Total number of sampling steps.
        self.prompt: str
        self.context: Tensor

    def predict_start_from_noise(self, noise_pred: Tensor, timestep: int, latent: Tensor) -> Tensor:
        return (
            torch.sqrt(1.0 / self.model.scheduler.alphas_cumprod[timestep]) * latent
            - torch.sqrt(1.0 / self.model.scheduler.alphas_cumprod[timestep] - 1) * noise_pred
        )

    @torch.no_grad()
    def init_prompt(self, prompt: str) -> None:
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
    def encode_tensor_to_latent(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            image = (image / 0.5 - 1).to(self.model.device)
            latents = self.model.vae.encode(image)["latent_dist"].sample()
            latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def decode_tensor_to_latent(self, latents: Tensor) -> Tensor:
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def one_step_dissolve(self, latent: Tensor, i: int) -> Tensor:
        _, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        # NOTE: This implementation use a reversed timesteps but can reach to
        # a stable dissolving effect.
        t = self.num_ddim_steps - self.model.scheduler.timesteps[i]
        latent = self.model.scheduler.scale_model_input(latent, t)
        cond_embeddings = cond_embeddings.repeat(latent.size(0), 1, 1)
        noise_pred = self.model.unet(latent, t, cond_embeddings).sample
        pred_x0 = self.predict_start_from_noise(noise_pred, t, latent)
        return pred_x0

    @torch.no_grad()
    def dissolve(self, image: Tensor, t: int) -> Tensor:
        self.init_prompt("")
        latent = self.encode_tensor_to_latent(image)
        ddim_latents = self.one_step_dissolve(latent, t)
        dissolved = self.decode_tensor_to_latent(ddim_latents)
        return dissolved


class StableDiffusionDissolving(ImageModule):
    r"""Perform dissolving transformation using StableDiffusion models.

    Based on :cite:`shi2024dissolving`, the dissolving transformation is essentially applying one-step
    reverse diffusion. Our implementation currently supports HuggingFace implementations of SD 1.4, 1.5
    and 2.1. SD 1.X tends to remove more details than SD2.1.

    .. list-table:: Title
        :widths: 32 32 32
        :header-rows: 1

        * - SD 1.4
          - SD 1.5
          - SD 2.1
        * - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.4.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-1.5.png
          - figure:: https://raw.githubusercontent.com/kornia/data/main/dslv-sd-2.1.png

    Args:
        version: the version of the stable diffusion model.
        **kwargs: additional arguments for `.from_pretrained`.
    """

    def __init__(self, version: str = "2.1", **kwargs: Any):
        super().__init__()
        StableDiffusionPipeline = diffusers.StableDiffusionPipeline
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

        if version == "1.4":
            self._sdm_model = StableDiffusionPipeline.from_pretrained(  # type:ignore
                "CompVis/stable-diffusion-v1-4", scheduler=scheduler, **kwargs
            )
        elif version == "1.5":
            self._sdm_model = StableDiffusionPipeline.from_pretrained(  # type:ignore
                "runwayml/stable-diffusion-v1-5", scheduler=scheduler, **kwargs
            )
        elif version == "2.1":
            self._sdm_model = StableDiffusionPipeline.from_pretrained(  # type:ignore
                "stabilityai/stable-diffusion-2-1", scheduler=scheduler, **kwargs
            )
        else:
            raise NotImplementedError

        self.model = _DissolvingWraper_HF(self._sdm_model, num_ddim_steps=1000)

    def forward(self, input: Tensor, step_number: int) -> Tensor:
        return self.model.dissolve(input, step_number)
