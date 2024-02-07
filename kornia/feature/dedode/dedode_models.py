import torch
from torch import nn

from .decoder import ConvRefiner, Decoder
from .descriptor import DeDoDeDescriptor
from .detector import DeDoDeDetector
from .encoder import VGG19, VGG_DINOv2


def dedode_detector_L():
    NUM_PROTOTYPES = 1
    residual = True
    hidden_blocks = 8
    amp_dtype = torch.float16
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


def dedode_descriptor_B():
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16
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


def dedode_descriptor_G():
    NUM_PROTOTYPES = 256  # == descriptor size
    residual = True
    hidden_blocks = 5
    amp_dtype = torch.float16
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
    vgg_kwargs = dict(amp=amp, amp_dtype=amp_dtype)
    dinov2_kwargs = dict(amp=amp, amp_dtype=amp_dtype, dinov2_weights=None)
    encoder = VGG_DINOv2(vgg_kwargs=vgg_kwargs, dinov2_kwargs=dinov2_kwargs)
    decoder = Decoder(conv_refiner, num_prototypes=NUM_PROTOTYPES)
    model = DeDoDeDescriptor(encoder=encoder, decoder=decoder)
    return model


get_detector = {
    "L": dedode_detector_L,
}
get_descriptor = {
    "B": dedode_descriptor_B,
    "G": dedode_descriptor_G,
}
