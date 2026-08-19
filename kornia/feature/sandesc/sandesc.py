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

from ._modules import UNetDownBlock, UNetUpBlock

# One checkpoint per supported keypoint detector. The descriptor weights
# are trained to pair with that detector.
urls: dict[str, list[str]] = {
    "aliked": [
        "https://huggingface.co/mattia-durso/SANDesc/resolve/main/pretrained/aliked/sandesc_aliked.pth",
        "https://raw.githubusercontent.com/mattiadurso/SANDesc/main/pretrained/aliked/sandesc_aliked.pth",
        "https://cloud.tugraz.at/index.php/s/Ww3t7b3ipnAoejS/download",
    ],
    "dedode": [
        "https://huggingface.co/mattia-durso/SANDesc/resolve/main/pretrained/dedode/sandesc_dedode.pth",
        "https://raw.githubusercontent.com/mattiadurso/SANDesc/main/pretrained/dedode/sandesc_dedode.pth",
        "https://cloud.tugraz.at/index.php/s/47Mcao9qydBppMB/download",
    ],
}

# `grid_sample` convention each detector's normalized keypoints follow. ALIKED normalizes with
# ``wh = [w-1, h-1]`` (align_corners=True: [-1, 1] maps to pixel centers 0 and w-1/h-1). DeDoDe
# normalizes with half-pixel centers (align_corners=False: [-1, 1] maps to the outer pixel edges).
detector_align_corners: dict[str, bool] = {
    "aliked": True,
    "dedode": False,
}


class SANDesc(nn.Module):
    r"""Module that computes dense local descriptors using the SANDesc method.

    See :cite:`durso2026sandesc` for details.

    SANDesc learns dense local descriptors for use with existing keypoint detectors. It uses a revised
    U-Net-like encoder-decoder enhanced with Convolutional Block Attention Modules and residual paths
    to produce a dense descriptor volume from an input image. The checkpoints returned by
    :meth:`from_pretrained` are trained per detector, so the descriptor must be paired with the
    detector it was trained for.

    .. image:: _static/img/SANDesc.png

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
        amp: If True, run :meth:`forward` under CUDA automatic mixed precision.
        amp_dtype: Autocast dtype used when ``amp`` is enabled (e.g. ``torch.float16``
            or ``torch.bfloat16``). AMP is scoped to CUDA; it is a no-op on CPU/MPS.
        keypoint_align_corners: ``align_corners`` convention used by :meth:`describe` to sample
            the descriptor volume at normalized keypoints. Must match the convention the keypoints
            were normalized with: ``True`` for ALIKED, ``False`` for DeDoDe. Set automatically by
            :meth:`from_pretrained`; can be overridden per call via ``describe(..., align_corners=...)``.

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
        keypoint_align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.keypoint_align_corners = keypoint_align_corners
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

        .. note::
            The pretrained checkpoints are released under a non-commercial license, which is
            more restrictive than Kornia's Apache-2.0 code license. Check the license terms at
            the checkpoint source before using the pretrained weights outside of research.

        Args:
            detector: Keypoint detector the descriptor was trained for. One of
                ``"aliked"`` or ``"dedode"``. Selects the default checkpoints and the
                ``align_corners`` convention (see :attr:`keypoint_align_corners`) used by
                :meth:`describe`, since that convention depends on the detector, not the
                checkpoint source.
            url: Direct URL to a checkpoint, or a list of URLs tried in order. If
                ``None``, the predefined URLs for ``detector`` are used.
            amp: If True, run :meth:`forward` under CUDA automatic mixed precision.
            amp_dtype: Autocast dtype used when ``amp`` is enabled.

        Returns:
            The SANDesc model with the pretrained weights loaded, in eval mode.
        """
        if detector not in detector_align_corners:
            raise ValueError(f"Unknown detector: {detector}. Available: {list(detector_align_corners)}")
        if url is None:
            url = urls[detector]

        model = cls(
            skip_connection=True,
            spatial_attention=True,
            third_block=True,
            amp=amp,
            amp_dtype=amp_dtype,
            keypoint_align_corners=detector_align_corners[detector],
        )
        state_dict = load_state_dict_from_url(
            url,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, img: Tensor, pad_if_not_divisible: bool = False) -> Tensor:
        """Compute the dense descriptor volume of the input images.

        Args:
            img: Input images of shape :math:`(B, C, H, W)`, with values in the
                :math:`[0, 1]` range. No further normalization is applied.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the
                closest 16-multiply and the output is cropped back to the input resolution.

        Returns:
            The dense descriptor volume of shape :math:`(B, D, H, W)`, where :math:`D` is the
            descriptor dimension.

        """
        KORNIA_CHECK_SHAPE(img, ["B", "C", "H", "W"])
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
        with torch.autocast(device_type, enabled=self.amp and device_type == "cuda", dtype=self.amp_dtype):
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

    def describe(
        self,
        images: Tensor,
        keypoints: Tensor | None = None,
        mode: str = "nearest",
        normalize: bool = True,
        pad_if_not_divisible: bool = False,
        align_corners: bool | None = None,
    ) -> Tensor:
        """Describe keypoints in the input images. If keypoints are not provided, returns the dense descriptors.

        The images are passed through :meth:`forward` to obtain a dense descriptor
        volume, which is then sampled at the keypoints with ``grid_sample``.

        Args:
            images: Input images of shape :math:`(B, C, H, W)`, with values in the
                :math:`[0, 1]` range. No further normalization is applied.
            keypoints: An optional tensor of shape :math:`(B, N, 2)` containing the detected
                keypoints, normalized to the :math:`[-1, 1]` range. The normalization convention
                must match ``align_corners``: kornia ALIKED keypoints use ``align_corners=True``,
                kornia DeDoDe keypoints use ``align_corners=False``.
            mode: ``grid_sample`` interpolation mode, ``"nearest"`` (default) or
                ``"bilinear"``.
            normalize: If True (default), L2-normalize the descriptors.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the
                closest 16-multiply.
            align_corners: ``grid_sample`` convention to sample ``keypoints`` with. If ``None``
                (default), uses :attr:`keypoint_align_corners`.

        Returns:
            The descriptors of shape :math:`(B, N, D)`, or the dense descriptor volume of
            shape :math:`(B, D, H, W)` when ``keypoints`` is None. :math:`D` is the descriptor
            dimension.

        """
        volume = self.forward(images, pad_if_not_divisible=pad_if_not_divisible)
        if keypoints is None:
            return F.normalize(volume, p=2, dim=1) if normalize else volume

        KORNIA_CHECK_SHAPE(keypoints, ["B", "N", "2"])
        if align_corners is None:
            align_corners = self.keypoint_align_corners
        # grid_sample does not support half/bfloat16 (autocast) volumes; upcast those to
        # float32 and match the grid dtype to the volume to avoid a dtype mismatch.
        sample_volume = volume.float() if volume.dtype in (torch.float16, torch.bfloat16) else volume
        grid = keypoints[:, None].to(device=sample_volume.device, dtype=sample_volume.dtype)
        sampled = F.grid_sample(sample_volume, grid, mode=mode, align_corners=align_corners)
        descriptors = sampled[:, :, 0].mT  # B,N,des_dim
        return F.normalize(descriptors, p=2, dim=-1) if normalize else descriptors
