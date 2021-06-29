from typing import Dict

import torch
import torch.nn as nn
import torchvision.models

urls: Dict[str, str] = {}
urls["defmo_encoder"] = "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/encoder_best.pt"
urls["defmo_rendering"] = "http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/rendering_best.pt"

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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
            Bottleneck(1024, 256),
            nn.PixelShuffle(2),
            Bottleneck(256, 64),
            nn.PixelShuffle(2),
            Bottleneck(64, 16),
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
        shuffled_times = torch.stack(shuffled_times, 1).contiguous().transpose()
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
