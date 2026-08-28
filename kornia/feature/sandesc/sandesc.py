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

"""SANDesc dense descriptor network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.core.download import load_state_dict_from_url
from kornia.geometry.conversions import normalize_pixel_coordinates

from ._modules import UNetDownBlock, UNetUpBlock

# Descriptor checkpoint, trained to pair with the ALIKED detector, served from mirrors
# holding the same file (sha256 0805481bd7b40672ee5fe07343904c45937f4cadc28ff52a01b1dd401d584808).
# The primary URL's trailing filename is also the hub cache filename; its query string is
# dropped by the cache-name resolution.
urls: list[str] = [
    "https://huggingface.co/mattia-durso/SANDesc/resolve/main/pretrained/sandesc_aliked.pth?download=true",
    "https://cloud.tugraz.at/index.php/s/dBiF999GBMoRg8w/download/sandesc_aliked.pth",
]


class SANDesc(nn.Module):
    r"""Module that computes dense local descriptors using the SANDesc method.

    See :cite:`durso2026sandesc` for details.

    SANDesc learns dense local descriptors for use with an existing keypoint detector. It uses a revised
    U-Net-like encoder-decoder enhanced with Convolutional Block Attention Modules and residual paths
    to produce a dense descriptor volume from an input image, which is then sampled at the keypoints.
    The checkpoint returned by :meth:`from_pretrained` is trained to pair with the ALIKED detector.

    .. note::
        :cite:`durso2026sandesc` reports improved matching performance over ALIKED descriptors at
        large keypoint budgets (e.g. 8k keypoints per image) and high resolutions (e.g. 4K). Those
        numbers come from the paper; this module has not been benchmarked inside kornia.

    Args:
        ch_in: Number of input channels.
        kernel_size: Kernel size of the convolutional layers.
        activation: Activation function: ``'relu'`` or ``'gelu'``.
        norm: Normalization layer type: ``'batch'``, ``'instance'`` or ``'group'``.
        skip_connection: If True, add a residual path and a second unet block inside each down
            and up block. The encoder-to-decoder concatenations are always applied.
        spatial_attention: If True, add spatial attention to the network. Requires
            ``skip_connection=True``.
        third_block: If True, add a third unet block to the network. Requires
            ``skip_connection=True``.
        down_output_channels: Output channels of each down block, 5 elements.
        up_output_channels: Output channels of each up block, 4 elements. The last element is the
            descriptor dimension. Add +1 to the last element to match the DISK unet,
            e.g. ``[64, 64, 64, 128 + 1]``.
        amp: If True, run :meth:`extract_dense_map` under CUDA automatic mixed precision.
        amp_dtype: Autocast dtype used when ``amp`` is enabled (e.g. ``torch.float16``
            or ``torch.bfloat16``). AMP is scoped to CUDA; it is a no-op on CPU/MPS.

    Example:
        >>> sandesc = SANDesc.from_pretrained()
        >>> images = torch.rand(1, 3, 64, 64)
        >>> keypoints = torch.rand(1, 10, 2) * 63  # pixel coordinates [x, y], from any detector
        >>> descriptors = sandesc(images, keypoints)
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
    ) -> None:
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
            down_output_channels[-1],
            down_output_channels[-2],
            up_output_channels[0],
            **common,
        )
        self.up1 = UNetUpBlock(
            up_output_channels[0],
            down_output_channels[-3],
            up_output_channels[1],
            **common,
        )
        self.up2 = UNetUpBlock(
            up_output_channels[1],
            down_output_channels[-4],
            up_output_channels[2],
            **common,
        )
        self.up3 = UNetUpBlock(
            up_output_channels[2],
            down_output_channels[-5],
            up_output_channels[3],
            kernel_size=kernel_size,
            activation=None,
            norm=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        url: str | list[str] | None = None,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> SANDesc:
        """Load the SANDesc descriptor weights trained to pair with the ALIKED detector.

        Weights are fetched with :func:`kornia.core.download.load_state_dict_from_url`,
        which caches the file locally (under the torch hub cache), only downloading it
        when it is missing. The weights are mapped to CPU; call ``.to(device)`` on the
        returned model to move it.

        .. note::
            The pretrained checkpoints are released under a non-commercial license, which is
            more restrictive than Kornia's Apache-2.0 code license. Check the license terms at
            the checkpoint source before using the pretrained weights outside of research.

        Args:
            url: Direct URL to a checkpoint, or a list of URLs tried in order. If
                ``None``, the predefined :data:`urls` are used.
            amp: If True, run :meth:`extract_dense_map` under CUDA automatic mixed precision.
            amp_dtype: Autocast dtype used when ``amp`` is enabled.

        Returns:
            The SANDesc model with the pretrained weights loaded, in eval mode.
        """
        if url is None:
            url = urls

        checkpoint = load_state_dict_from_url(
            url,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        # The released checkpoint wraps the weights together with the training config.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        # Build the architecture from the checkpoint's own config so a revised checkpoint
        # loads into the right network; missing keys fall back to the released config.
        config = checkpoint.get("config", {}).get("model", {})
        model = cls(
            ch_in=config.get("unet_ch_in", 3),
            kernel_size=config.get("unet_kernel_size", 5),
            activation=config.get("unet_activ", "gelu"),
            norm=config.get("unet_norm", "batch"),
            skip_connection=config.get("unet_with_skip_connections", True),
            spatial_attention=config.get("unet_spatial_attention", True),
            third_block=config.get("third_block", True),
            amp=amp,
            amp_dtype=amp_dtype,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def extract_dense_map(self, img: Tensor, pad_if_not_divisible: bool = False) -> Tensor:
        """Compute the dense descriptor volume of the input images.

        Args:
            img: Input images of shape :math:`(B, C, H, W)`, with values in the :math:`[0, 1]`
                range and :math:`C` equal to ``ch_in``. No further normalization is applied.
                Grayscale inputs are rejected rather than replicated to 3 channels: the pretrained
                weights were trained on RGB, so the caller decides whether that substitution is
                acceptable and applies :func:`kornia.color.grayscale_to_rgb` themselves.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the
                closest 16-multiply and the output is cropped back to the input resolution.

        Returns:
            The dense descriptor volume of shape :math:`(B, D, H, W)`, where :math:`D` is the
            descriptor dimension.

        """
        KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
        ch_in = self.conv_highest.in_channels
        if img.size(1) != ch_in:
            raise ValueError(
                f"Expected {ch_in} feature channels in input, got {img.size(1)}. Grayscale inputs "
                "are not converted, since the pretrained weights were trained on RGB; convert them "
                "explicitly with kornia.color.grayscale_to_rgb if that is what you want."
            )
        h, w = img.shape[-2:]
        if pad_if_not_divisible:
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            img = F.pad(img, (0, pd_w, 0, pd_h), value=0.0)
        elif h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"Image height and width must be multiples of 16, got {h}x{w}. "
                "Use pad_if_not_divisible=True to zero-pad the input."
            )

        # AMP is intentionally scoped to CUDA only: float16 autocast is unsupported on CPU and a no-op on MPS.
        device_type = "cuda" if img.device.type == "cuda" else "cpu"
        with torch.autocast(
            device_type,
            enabled=self.amp and device_type == "cuda",
            dtype=self.amp_dtype,
        ):
            x0 = self.conv_highest(img)  # B,c_in,H,W

            x1 = self.down0(x0)  # B,C1,H/2,W/2
            x2 = self.down1(x1)  # B,C2,H/4,W/4
            x3 = self.down2(x2)  # B,C3,H/8,W/8
            x4 = self.down3(x3)  # B,C4,H/16,W/16

            x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
            x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
            x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
            x8 = self.up3(x7, x0)  # B,des_dim,H,W

        return x8[..., :h, :w]

    def forward(
        self,
        images: Tensor,
        keypoints: Tensor,
        mode: str = "bilinear",
        normalize: bool = True,
        pad_if_not_divisible: bool = False,
        return_volume: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Describe keypoints given in pixel coordinates.

        SANDesc does not bundle a detector: run any keypoint detector and pass its pixel
        coordinates here. The images are passed through :meth:`extract_dense_map` and the
        resulting descriptor volume is sampled at the keypoints with ``grid_sample``.

        Args:
            images: Input images of shape :math:`(B, C, H, W)`, with values in the :math:`[0, 1]`
                range and :math:`C` equal to ``ch_in``. No further normalization is applied.
            keypoints: Keypoints in pixel coordinates of shape :math:`(B, N, 2)` as ``[x, y]``,
                e.g. the output of a kornia detector such as ALIKED or DISK. Keypoints outside
                the image sample the zero padding and come back as zero descriptors.
            mode: ``grid_sample`` interpolation mode, ``"bilinear"`` (default) or ``"nearest"``.
            normalize: If True (default), L2-normalize the descriptors after sampling.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the
                closest 16-multiply before the descriptor volume is computed.
            return_volume: If True, also return the dense descriptor volume of shape
                :math:`(B, D, H, W)`, L2-normalized over the channel dimension when
                ``normalize`` is True.

        Returns:
            The descriptors of shape :math:`(B, N, D)` in the dtype of ``images`` (sampling
            happens in float32 for half-precision volumes), where :math:`D` is the descriptor
            dimension. With ``return_volume=True``, a tuple of the descriptors and the dense
            descriptor volume.

        """
        KORNIA_CHECK_SHAPE(keypoints, ["B", "N", "2"])
        volume = self.extract_dense_map(images, pad_if_not_divisible=pad_if_not_divisible)
        height, width = images.shape[-2:]
        # normalize_pixel_coordinates divides by (side - 1), which is the
        # ``align_corners=True`` convention grid_sample is called with below.
        grid = normalize_pixel_coordinates(keypoints, height, width)[:, None]
        # grid_sample does not support half/bfloat16 (autocast) volumes; upcast those to
        # float32 and match the grid dtype to the volume to avoid a dtype mismatch.
        sample_volume = volume.float() if volume.dtype in (torch.float16, torch.bfloat16) else volume
        grid = grid.to(device=sample_volume.device, dtype=sample_volume.dtype)
        sampled = F.grid_sample(sample_volume, grid, mode=mode, align_corners=True)
        descriptors = sampled[:, :, 0].mT  # B,N,des_dim
        if normalize:
            descriptors = F.normalize(descriptors, p=2, dim=-1)
        if return_volume:
            # the descriptor dimension of the (B, D, H, W) volume is dim=1
            volume = F.normalize(volume, p=2, dim=1) if normalize else volume
            return descriptors.to(images.dtype), volume.to(images.dtype)
        return descriptors.to(images.dtype)
