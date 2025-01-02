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
from torch import nn

from .decoder import ConvRefiner, Decoder
from .descriptor import DeDoDeDescriptor
from .detector import DeDoDeDetector
from .encoder import VGG19, VGG_DINOv2


def dedode_detector_L(amp_dtype: torch.dtype = torch.float16) -> DeDoDeDetector:
    """Get DeDoDe descriptor of type L."""
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                128,
                64 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 64,
                64,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner)
    model = DeDoDeDetector(encoder=encoder, decoder=decoder)
    return model


def dedode_descriptor_B(amp_dtype: torch.dtype = torch.float16) -> DeDoDeDescriptor:
    """Get DeDoDe descriptor of type B."""
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG19(amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder)
    return model


def dedode_descriptor_G(amp_dtype: torch.dtype = torch.float16) -> DeDoDeDescriptor:
    """Get DeDoDe descriptor of type G."""
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
                512,
                256 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + NUM_PROTOTYPES,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    vgg_kwargs = {"amp": amp, "amp_dtype": amp_dtype}
    dinov2_kwargs = {"amp": amp, "amp_dtype": amp_dtype, "dinov2_weights": None}
    encoder = VGG_DINOv2(vgg_kwargs=vgg_kwargs, dinov2_kwargs=dinov2_kwargs)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder)
    return model


def get_detector(kind: str = "L", amp_dtype: torch.dtype = torch.float16) -> DeDoDeDetector:
    """Get DeDoDe detector."""
    if kind == "L":
        return dedode_detector_L(amp_dtype)
    raise ValueError(f"Unknown detector kind: {kind}")


def get_descriptor(kind: str = "B", amp_dtype: torch.dtype = torch.float16) -> DeDoDeDescriptor:
    """Get DeDoDe descriptor."""
    if kind == "B":
        return dedode_descriptor_B(amp_dtype)
    if kind == "G":
        return dedode_descriptor_G(amp_dtype)
    raise ValueError(f"Unknown descriptor kind: {kind}")
