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

from kornia.color.ycbcr import RgbToYcbcr, YcbcrToRgb
from kornia.config import kornia_config
from kornia.onnx.download import CachedDownloader

url = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"


class SmallSRNet(nn.Module):
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
            with torch.no_grad():
                weight_init(self)

    def load_from_file(self, path_file: str) -> None:
        """Load pretrained super-resolution weights from the model cache.

        Args:
            path_file: Checkpoint URL or path passed to the cached downloader.
                The downloaded state dictionary is loaded strictly and the
                module is switched to evaluation mode.
        """
        # use torch.hub to load pretrained model
        model_path = CachedDownloader.download_to_cache(
            path_file, "small_sr.pth", download=True, suffix=".pth", cache_dir=kornia_config.hub_onnx_dir
        )
        pretrained_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a single-channel image tensor.

        Args:
            x: Luminance image tensor with shape :math:`(B, 1, H, W)`, where
                :math:`B` is batch size, :math:`H` is height, and :math:`W` is
                width.

        Returns:
            Upscaled luminance tensor with shape
            :math:`(B, 1, H * r, W * r)`, where ``r`` is the configured
            upscale factor.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


def weight_init(module: nn.Module) -> None:
    """Initialize model weights for Conv2d layers."""
    if isinstance(module, nn.Conv2d):
        # Use orthogonal initialization with gain for all conv layers
        # conv4 (the last layer before pixel shuffle) uses default gain
        if module.out_channels in {64, 32}:  # conv1, conv2, conv3
            torch.nn.init.orthogonal_(module.weight, torch.nn.init.calculate_gain("relu"))
        else:  # conv4 (upscale_factor**2 channels)
            torch.nn.init.orthogonal_(module.weight)


class SmallSRNetWrapper(nn.Module):
    """Wrap a Super-Resolution model with pre-processing and post-processing.

    Args:
        upscale_factor: The factor by which the image resolution is increased.
        pretrained: Whether to load weights from a pre-trained model. Default: True.
    """

    def __init__(self, upscale_factor: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        self.rgb_to_ycbcr = RgbToYcbcr()
        self.ycbcr_to_rgb = YcbcrToRgb()
        self.model = SmallSRNet(upscale_factor=upscale_factor, pretrained=pretrained)
        self.upscale_factor = upscale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply RGB super-resolution through a luminance-only SR model.

        Args:
            input: RGB image tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            RGB tensor upscaled by ``self.upscale_factor``. The Y channel is
            predicted by the super-resolution model, while Cb and Cr channels
            are resized with bicubic interpolation before conversion back to
            RGB.
        """
        ycbcr = self.rgb_to_ycbcr(input)
        y, cb, cr = ycbcr.split(1, dim=1)
        out_y = self.model(y)
        out_cb = torch.nn.functional.interpolate(cb, scale_factor=self.upscale_factor, mode="bicubic")
        out_cr = torch.nn.functional.interpolate(cr, scale_factor=self.upscale_factor, mode="bicubic")
        out = torch.cat([out_y, out_cb, out_cr], dim=1)
        return self.ycbcr_to_rgb(out)
