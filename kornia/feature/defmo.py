from typing import Dict

import torch
import torch.nn as nn
import torchvision.models

urls: Dict[str, str] = {}
urls["defmo_encoder"] = "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/encoder_best.pt"
urls["defmo_rendering"] = "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/rendering_best.pt"


class EncoderDeFMO(nn.Module):
    def __init__(self):
        super(EncoderDeFMO, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        pretrained_weights = modelc1[0].weight
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
        self.net = nn.Sequential(modelc1, modelc2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class RenderingDeFMO(nn.Module):
    def __init__(self):
        super(RenderingDeFMO, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(2049, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            torchvision.models.resnet.Bottleneck(1024, 256),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(256, 64),
            nn.PixelShuffle(2),
            torchvision.models.resnet.Bottleneck(64, 16),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.net = model

    def forward(self, latent, times):
        renders = []
        shuffled_times = []
        for ki in range(times.shape[0]):
            shuffled_times.append(torch.randperm(times.shape[1]))
        shuffled_times = torch.stack(shuffled_times, 1).contiguous().T
        for ki in range(times.shape[1]):
            t_tensor = (
                times[range(times.shape[0]), shuffled_times[:, ki]]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, latent.shape[2], latent.shape[3])
            )
            latenti = torch.cat((t_tensor, latent), 1)
            result = self.net(latenti)
            renders.append(result)
        renders = torch.stack(renders, 1).contiguous()
        renders[:, :, :4] = torch.sigmoid(renders[:, :, :4])
        for ki in range(times.shape[0]):
            renders[ki, shuffled_times[ki, :]] = renders[ki, :].clone()
        return renders


class DeFMO(nn.Module):
    """
    Module that disentangle a fast-moving object from the background and performs deblurring.

    This is based on the original code from paper "DeFMO: Deblurring and Shape Recovery
        of Fast Moving Objects". See :cite:`DeFMO2021` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model. Default: false.
    Returns:
        Temporal super-resolution without background.
    Shape:
        - Input: (B, 6, H, W)
        - Output: (B, S, 4, H, W)

    Examples:
        >>> input = torch.rand(16, 6, 240, 320)
        >>> defmo = DeFMO()
        >>> tsr_nobgr = defmo(input) # 16x24x4x240x320
    """

    def __init__(self, pretrained: bool = False) -> None:
        super(DeFMO, self).__init__()
        self.tsr_steps: int = 24
        self.encoder = EncoderDeFMO()
        self.rendering = RenderingDeFMO()

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['defmo_encoder'], map_location=lambda storage, loc: storage
            )
            self.encoder.load_state_dict(pretrained_dict, strict=True)
            pretrained_dict_ren = torch.hub.load_state_dict_from_url(
                urls['defmo_rendering'], map_location=lambda storage, loc: storage
            )
            self.rendering.load_state_dict(pretrained_dict_ren, strict=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(input)
        times = torch.linspace(0, 1, self.tsr_steps).to(input.device)[None].repeat(input.shape[0], 1)
        x_out = self.rendering(latent, times)
        return x_out
