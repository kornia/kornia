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

"""SANDesc dense descriptor network.

SANDesc ("A Streamlined Attention-Based Network for Descriptor Extraction",
3DV 2026, https://arxiv.org/pdf/2601.13126) learns dense local descriptors for
use with existing keypoint detectors. It uses a revised U-Net-like encoder-decoder
enhanced with Convolutional Block Attention Modules and residual paths to
produce a dense descriptor volume from an input image.
SANDesc usually outperforms existing descriptors modules on high-resolution
images, while still fitting in 24GB of VRAM.


Example usage with ALIKED:

# Initialize ALIKED and extract points
aliked = ALIKED.from_pretrained(model_name="aliked-n16", max_num_keypoints=max_kpts, device=device)
with torch.inference_mode():
    feat_A = aliked(img_A[None])[0]
    feat_B = aliked(img_B[None])[0]

# Initialize SANDesc with the pretrained weights for ALIKED and describe the keypoints.
sandesc = SANDesc.from_pretrained(detector="aliked", amp=True).to(device).eval()
with torch.inference_mode():
    kpts_A_norm = normalize_pixel_coordinates(feat_A.keypoints, img_A.shape[-2], img_A.shape[-1])
    kpts_B_norm = normalize_pixel_coordinates(feat_B.keypoints, img_B.shape[-2], img_B.shape[-1])
    desc_A = sandesc.describe(img_A[None], kpts_A_norm[None])  # (1, N, des_dim)
    desc_B = sandesc.describe(img_B[None], kpts_B_norm[None])

"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.download import load_state_dict_from_url

from .modules import UNetDownBlock, UNetUpBlock

# One checkpoint per supported keypoint detector. The descriptor weights
# are trained to pair with that detector.
urls: dict[str, list[str]] = {
    "aliked": [
        "https://raw.githubusercontent.com/mattiadurso/SANDesc/main/pretrained/aliked/sandesc_aliked.pth",
        "https://cloud.tugraz.at/index.php/s/Ww3t7b3ipnAoejS/download",
    ],
    "dedode": [
        "https://raw.githubusercontent.com/mattiadurso/SANDesc/main/pretrained/dedode/sandesc_dedode.pth",
        "https://cloud.tugraz.at/index.php/s/47Mcao9qydBppMB/download",
    ],
}


class SANDesc(nn.Module):
    """UNet-style encoder-decoder producing a dense descriptor volume.

    Example:
        >>> sandesc = SANDesc().eval()
        >>> images = torch.rand(1, 3, 64, 64)
        >>> keypoints = torch.rand(1, 10, 2) * 2 - 1
        >>> descriptors = sandesc.describe(images, keypoints)
        >>> descriptors.shape
        torch.Size([1, 10, 128])

    """

    def __init__(
        self,
        ch_in: int = 3,
        kernel_size: int = 5,
        activation: str = "gelu",
        norm: str = "batch",
        skip_connection: bool = False,
        spatial_attention: bool = False,
        third_block: bool = False,
        down_output_channels: list[int] | None = None,
        up_output_channels: list[int] | None = None,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        **kwargs: object,
    ) -> None:
        """Build the descriptor network.

        The last element of ``up_output_channels`` is the descriptor dimension.

        Args:
            ch_in: Number of input channels.
            kernel_size: Kernel size of the convolutional layers.
            activation: Activation function: 'relu' or 'gelu'.
            norm: Normalization layer type.
            skip_connection: If True, add skip connections and a second unet
                block to the network.
            spatial_attention: If True, add spatial attention to the network.
            third_block: If True, add a third unet block to the network.
            down_output_channels: Output channels of each down block, 5 elements.
            up_output_channels: Output channels of each up block, 4 elements. Add +1
                to the last element to match the DISK unet, e.g. [64, 64, 64, 128 + 1].
            amp: If True, run :meth:`forward` under CUDA automatic mixed precision.
            amp_dtype: Autocast dtype used when ``amp`` is enabled (e.g. ``torch.float16``
                or ``torch.bfloat16``). AMP is scoped to CUDA; it is a no-op on CPU/MPS.
            **kwargs: Ignored extra keyword arguments.
        """
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        if down_output_channels is None:
            down_output_channels = [16, 32, 64, 64, 64]
        if up_output_channels is None:
            up_output_channels = [64, 64, 64, 128]
        if len(down_output_channels) != 5:
            raise ValueError(f"down_output_channels must have 5 elements, got {len(down_output_channels)}.")
        if len(up_output_channels) != 4:
            raise ValueError(f"up_output_channels must have 4 elements, got {len(up_output_channels)}.")
        self.conv_highest = nn.Conv2d(
            ch_in,
            down_output_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

        common = {
            "kernel_size": kernel_size,
            "activation": activation,
            "norm": norm,
            "skip_connection": skip_connection,  # and second block
            "spatial_attention": spatial_attention,
            "third_block": third_block,
        }

        self.down0 = UNetDownBlock(down_output_channels[0], down_output_channels[1], **common)
        self.down1 = UNetDownBlock(down_output_channels[1], down_output_channels[2], **common)
        self.down2 = UNetDownBlock(down_output_channels[2], down_output_channels[3], **common)
        self.down3 = UNetDownBlock(down_output_channels[3], down_output_channels[4], **common)

        self.up0 = UNetUpBlock(
            down_output_channels[-1] + down_output_channels[-2],
            up_output_channels[0],
            **common,
        )
        self.up1 = UNetUpBlock(
            down_output_channels[-3] + up_output_channels[0],
            up_output_channels[1],
            **common,
        )
        self.up2 = UNetUpBlock(
            down_output_channels[-4] + up_output_channels[1],
            up_output_channels[2],
            **common,
        )
        self.up3 = UNetUpBlock(
            down_output_channels[-5] + up_output_channels[2],
            up_output_channels[3],
            kernel_size=kernel_size,
            activation=None,
            norm=None,
        )

    def load_weights(self, weights: str) -> None:
        """Load weights into the model from a local state_dict file.

        Args:
            weights (str): Path to the weights file (a flat state_dict).

        """
        state_dict = torch.load(weights, weights_only=True)
        self.load_state_dict(state_dict)

    @classmethod
    def from_pretrained(
        cls,
        detector: str = "aliked",
        url: str | list[str] | None = None,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> SANDesc:
        """Load SANDesc descriptor weights trained for a given keypoint detector.

        Weights are fetched with :func:`kornia.core.download.load_state_dict_from_url`,
        which tries each source in turn and caches the file locally (under the torch
        hub cache), only downloading it when it is missing. The weights are mapped to
        CPU; call ``.to(device)`` on the returned model to move it.

        Args:
            detector: Keypoint detector the descriptor was trained for. One of
                ``"aliked"`` or ``"dedode"``. Selects the default checkpoints.
            url: Direct URL to a checkpoint, or a list of URLs tried in order. If
                ``None``, the predefined URLs for ``detector`` are used.
            amp: If True, run :meth:`forward` under CUDA automatic mixed precision.
            amp_dtype: Autocast dtype used when ``amp`` is enabled.

        Returns:
            The SANDesc model with the pretrained weights loaded, in eval mode.
        """
        if url is None:
            if detector not in urls:
                raise ValueError(f"Unknown detector: {detector}. Available: {list(urls)}")
            url = urls[detector]
            # The fallback host names every file generically ("download"), so force a
            # distinct cache filename per detector that is stable across both sources.
            file_name = f"sandesc_{detector}.pth"
        else:
            file_name = None

        model = cls(
            skip_connection=True,
            spatial_attention=True,
            third_block=True,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        state_dict = load_state_dict_from_url(
            url,
            map_location=torch.device("cpu"),
            file_name=file_name,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, img: Tensor) -> Tensor:
        """Compute the dense descriptor volume [B, des_dim, H, W] for input image."""
        KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
        h, w = img.shape[-2:]
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Image height and width must be multiples of 16, got {h}x{w}.")

        # AMP is scoped to "cuda": float16 autocast is unsupported on CPU and a no-op on MPS.
        with torch.autocast("cuda", enabled=self.amp, dtype=self.amp_dtype):
            x0 = self.conv_highest(img)  # B,c_in,H,W

            x1 = self.down0(x0)  # B,C1,H/2,W/2
            x2 = self.down1(x1)  # B,C2,H/4,W/4
            x3 = self.down2(x2)  # B,C3,H/8,W/8
            x4 = self.down3(x3)  # B,C4,H/16,W/16

            x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
            x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
            x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
            x8 = self.up3(x7, x0)  # B,des_dim,H,W

        return x8

    def describe(
        self,
        images: Tensor,
        keypoints: Tensor,
        return_desc_volume: bool = False,
        mode: str = "nearest",
        normalize: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample descriptors at the given keypoints.

        The images are passed through :meth:`forward` to obtain a dense descriptor
        volume, which is then sampled at the keypoints with ``grid_sample``.

        Args:
            images: Input images of shape :math:`(B, C, H, W)`.
            keypoints: Keypoints of shape :math:`(B, N, 2)`, normalized to the
                :math:`[-1, 1]` range (the convention used by the kornia ALIKED
                and DeDoDe detectors).
            return_desc_volume: If True, also return the dense descriptor volume of
                shape :math:`(B, des_dim, H, W)`.
            mode: ``grid_sample`` interpolation mode, ``"nearest"`` (default) or
                ``"bilinear"``.
            normalize: If True (default), L2-normalize the sampled descriptors.

        Returns:
            The descriptors of shape :math:`(B, N, des_dim)` (L2-normalized when
            ``normalize`` is True), or a tuple ``(descriptors, volume)`` when
            ``return_desc_volume`` is True.
        """
        KORNIA_CHECK_SHAPE(keypoints, ["B", "N", "2"])
        volume = self.forward(images)
        # grid_sample does not support half/bfloat16 (amp) volumes; upcast those to
        # float32 and match the grid dtype to the volume to avoid a dtype mismatch.
        sample_volume = volume.float() if volume.dtype in (torch.float16, torch.bfloat16) else volume
        grid = keypoints[:, None].to(sample_volume.dtype)
        sampled = F.grid_sample(sample_volume, grid, mode=mode, align_corners=False)
        descriptors = sampled[:, :, 0].mT  # B,N,des_dim
        if normalize:
            descriptors = F.normalize(descriptors, p=2, dim=-1)
            if return_desc_volume:
                volume = F.normalize(volume, p=2, dim=1)
        if return_desc_volume:
            return descriptors, volume
        return descriptors
