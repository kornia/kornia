from typing import Optional

import torch
from torch import nn

from kornia.color.ycbcr import RgbToYcbcr, YcbcrToRgb
from kornia.config import kornia_config
from kornia.core import Module, Tensor, concatenate
from kornia.models.utils import ResizePreProcessor
from kornia.utils.download import CachedDownloader

from .base import SuperResolution

url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"


class SmallSRNet(Module):
    """A small super-resolution model.

    This model uses the efficient sub-pixel convolution layer described in
    "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    - Shi et al for increasing the resolution of an image by an upscale factor.
    Taken from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html.
    """

    def __init__(self, upscale_factor: int, inplace: bool = False, pretrained: bool = True) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        if pretrained:
            self.load_from_file(url)
        else:
            self.apply(weight_init)

    def load_from_file(self, path_file: str) -> None:
        # use torch.hub to load pretrained model
        model_path = CachedDownloader.download_to_cache(
            path_file, "small_sr.pth", download=True, suffix=".pth", cache_dir=kornia_config.hub_onnx_dir
        )
        pretrained_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


def weight_init(model: Module) -> None:
    torch.nn.init.orthogonal_(model.conv1.weight, torch.nn.init.calculate_gain("relu"))
    torch.nn.init.orthogonal_(model.conv2.weight, torch.nn.init.calculate_gain("relu"))
    torch.nn.init.orthogonal_(model.conv3.weight, torch.nn.init.calculate_gain("relu"))
    torch.nn.init.orthogonal_(model.conv4.weight)


class SmallSRNetWrapper(Module):
    def __init__(self, upscale_factor: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.rgb_to_ycbcr = RgbToYcbcr()
        self.ycbcr_to_rgb = YcbcrToRgb()
        self.model = SmallSRNet(upscale_factor=upscale_factor, pretrained=pretrained)
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        ycbcr = self.rgb_to_ycbcr(input)
        y, cb, cr = ycbcr.split(1, dim=1)
        out_y = self.model(y)
        out_cb = torch.nn.functional.interpolate(cb, scale_factor=self.upscale_factor, mode="bicubic")
        out_cr = torch.nn.functional.interpolate(cr, scale_factor=self.upscale_factor, mode="bicubic")
        out = concatenate([out_y, out_cb, out_cr], dim=1)
        return self.ycbcr_to_rgb(out)


class SmallSRBuilder:
    @staticmethod
    def build(
        model_name: str = "small_sr", pretrained: bool = True, upscale_factor: int = 3, image_size: Optional[int] = None
    ) -> SuperResolution:
        if model_name.lower() == "small_sr":
            model = SmallSRNetWrapper(upscale_factor, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'small_sr'.")

        sr = SuperResolution(
            model,
            pre_processor=ResizePreProcessor(224, 224),
            post_processor=nn.Identity(),
            name=model_name,
        )
        if image_size is None:
            sr.pseudo_image_size = 224
        else:
            sr.input_image_size = image_size
            sr.output_image_size = image_size * 3
            sr.pseudo_image_size = image_size
        return sr
