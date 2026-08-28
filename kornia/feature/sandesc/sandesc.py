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
from kornia.feature.aliked import ALIKED
from kornia.geometry.conversions import normalize_pixel_coordinates

from ._modules import UNetDownBlock, UNetUpBlock

# Descriptor checkpoint, trained to pair with the ALIKED detector. The trailing
# filename makes Nextcloud name the download, which is also the hub cache filename.
urls: list[str] = [
    "https://cloud.tugraz.at/index.php/s/dBiF999GBMoRg8w/download/sandesc_aliked.pth",
]

# ALIKED normalizes keypoints with ``wh = [w-1, h-1]``, i.e. the ``grid_sample``
# ``align_corners=True`` convention: [-1, 1] maps to the pixel centers 0 and w-1/h-1.
_ALIGN_CORNERS: bool = True


def _build_detector(num_keypoints: int, pretrained: bool) -> nn.Module:
    """Build the ALIKED keypoint detector SANDesc is paired with, without its descriptor head.

    The variant is the one SANDesc was trained against, ``aliked-n16rot``; its checkpoint is
    resolved by :meth:`ALIKED.from_pretrained`.

    ALIKED is put in top-k mode (``detection_threshold=0``) so that it returns exactly
    ``num_keypoints`` keypoints per image; its threshold mode returns a variable count and
    could not be stacked into the batched output of :meth:`SANDesc.forward`.
    """
    kwargs = {
        "max_num_keypoints": num_keypoints,
        "detection_threshold": 0.0,
        "disable_descriptors": True,
    }
    if pretrained:
        return ALIKED.from_pretrained(model_name="aliked-n16rot", **kwargs)
    return ALIKED(model_name="aliked-n16rot", **kwargs)


class SANDesc(nn.Module):
    r"""Module that computes dense local descriptors using the SANDesc method.

    See :cite:`durso2026sandesc` for details.

    SANDesc learns dense local descriptors for use with an existing keypoint detector. It uses a revised
    U-Net-like encoder-decoder enhanced with Convolutional Block Attention Modules and residual paths
    to produce a dense descriptor volume from an input image, which is then sampled at the keypoints.
    The checkpoint returned by :meth:`from_pretrained` is trained to pair with the ALIKED detector.

    .. image:: _static/img/SANDesc.png

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
        keypoint_detector: If True, build the ALIKED detector as a submodule so that
            :meth:`forward` can run the full detect-and-describe pipeline. It is built with its
            descriptor head disabled, since SANDesc provides the descriptors. If False (default),
            no detector is built and only :meth:`describe` and :meth:`extract_dense_map` are
            available.
        num_keypoints: Number of keypoints :meth:`forward` asks the detector for. ALIKED fixes
            this at construction time, so it is a constructor argument here; a fixed count is
            also what makes the batched ``(B, N, ...)`` outputs of :meth:`forward` well defined.
            Must not exceed :math:`H \times W` of the images it is later run on.
        amp: If True, run :meth:`extract_dense_map` under CUDA automatic mixed precision.
        amp_dtype: Autocast dtype used when ``amp`` is enabled (e.g. ``torch.float16``
            or ``torch.bfloat16``). AMP is scoped to CUDA; it is a no-op on CPU/MPS.
        keypoint_align_corners: ``align_corners`` convention used by :meth:`describe` to sample
            the descriptor volume at normalized keypoints. Must match the convention the keypoints
            were normalized with; the default matches ALIKED. Pass ``False`` for keypoints
            normalized with half-pixel centers, e.g. kornia DeDoDe's. Can also be overridden per
            call via ``describe(..., align_corners=...)``.

    Example:
        >>> sandesc = SANDesc().eval()
        >>> images = torch.rand(1, 3, 64, 64)
        >>> keypoints = torch.rand(1, 10, 2) * 2 - 1
        >>> descriptors = sandesc.describe(images, keypoints)
        >>> descriptors.shape
        torch.Size([1, 10, 128])

        Detect and describe in one call:

        >>> sandesc = SANDesc.from_pretrained()  # doctest: +SKIP
        >>> keypoints, scores, descriptors = sandesc(images)  # doctest: +SKIP

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
        keypoint_detector: bool = False,
        num_keypoints: int = 2048,
        amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        keypoint_align_corners: bool = _ALIGN_CORNERS,
    ) -> None:
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.keypoint_align_corners = keypoint_align_corners
        self.num_keypoints = num_keypoints
        self.keypoint_detector: nn.Module | None = None
        if keypoint_detector:
            self.keypoint_detector = _build_detector(num_keypoints, pretrained=False)
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
        load_detector: bool = True,
        num_keypoints: int = 2048,
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
            load_detector: If True (default), also build the ALIKED detector with its own
                pretrained weights, so that the returned model is ready for :meth:`forward`.
                Pass False to load the descriptor alone -- it skips the detector checkpoint
                download, and leaves only :meth:`describe` and :meth:`extract_dense_map` usable.
            num_keypoints: Number of keypoints the detector is built for, see :class:`SANDesc`.
                Ignored when ``load_detector`` is False.

        Returns:
            The SANDesc model with the pretrained weights loaded, in eval mode.
        """
        if url is None:
            url = urls

        model = cls(
            skip_connection=True,
            spatial_attention=True,
            third_block=True,
            amp=amp,
            amp_dtype=amp_dtype,
            keypoint_align_corners=_ALIGN_CORNERS,
        )
        checkpoint = load_state_dict_from_url(
            url,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        # The released checkpoint wraps the weights together with the training config.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        # Loaded before the detector is attached: the checkpoint only holds descriptor weights,
        # so a registered detector submodule would make this strict load fail on missing keys.
        model.load_state_dict(state_dict)
        if load_detector:
            model.keypoint_detector = _build_detector(num_keypoints, pretrained=True)
            model.num_keypoints = num_keypoints
        model.eval()
        return model

    def extract_dense_map(self, img: Tensor, pad_if_not_divisible: bool = False) -> Tensor:
        """Compute the dense descriptor volume of the input images.

        Args:
            img: Input images of shape :math:`(B, C, H, W)`, with values in the :math:`[0, 1]`
                range and :math:`C` equal to ``ch_in``. No further normalization is applied.
                Grayscale inputs are rejected rather than replicated to 3 channels: the pretrained
                weights were trained on RGB, so the caller decides whether that substitution is
                acceptable and applies :func:`kornia.color.grayscale_to_rgb` themselves. The
                bundled ALIKED detector does convert them, so :meth:`forward` still requires the
                caller to convert first.
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

        The images are passed through :meth:`extract_dense_map` to obtain a dense descriptor
        volume, which is then sampled at the keypoints with ``grid_sample``.

        Args:
            images: Input images of shape :math:`(B, C, H, W)`, with values in the :math:`[0, 1]`
                range and :math:`C` equal to ``ch_in``. No further normalization is applied.
            keypoints: An optional tensor of shape :math:`(B, N, 2)` containing the detected
                keypoints, normalized to the :math:`[-1, 1]` range. The normalization convention
                must match ``align_corners``: kornia ALIKED keypoints use ``align_corners=True``,
                keypoints normalized with half-pixel centers use ``align_corners=False``.
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
        volume = self.extract_dense_map(images, pad_if_not_divisible=pad_if_not_divisible)
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

    def forward(
        self,
        images: Tensor,
        mode: str = "nearest",
        normalize: bool = True,
        pad_if_not_divisible: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Detect keypoints with ALIKED and describe them with SANDesc.

        Requires the model to have been built with ``keypoint_detector=True`` (or loaded via
        ``from_pretrained(load_detector=True)``). The detector runs with its own descriptor head
        disabled, so the descriptors returned here always come from SANDesc.

        Both models receive the raw ``[0, 1]`` images: each applies its own normalization and
        padding internally, so no preprocessed tensor is shared between them.

        .. note::
            To pair SANDesc with a detector it does not build itself, run that detector and pass
            its normalized keypoints to :meth:`describe`.

        Args:
            images: Input images of shape :math:`(B, C, H, W)`, with values in the :math:`[0, 1]`
                range and :math:`C` equal to ``ch_in``.
            mode: ``grid_sample`` interpolation mode, ``"nearest"`` (default) or ``"bilinear"``.
            normalize: If True (default), L2-normalize the descriptors.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the
                closest 16-multiply before the descriptor volume is computed.

        Returns:
            A tuple of keypoints in pixel coordinates :math:`(B, N, 2)` as ``[x, y]``, their
            detection scores :math:`(B, N)`, and the descriptors :math:`(B, N, D)`.

        """
        if self.keypoint_detector is None:
            raise RuntimeError(
                "SANDesc has no keypoint detector. Build it with "
                "SANDesc(keypoint_detector=True), or with "
                "SANDesc.from_pretrained(load_detector=True)."
            )
        height, width = images.shape[-2:]
        if self.num_keypoints > height * width:
            raise ValueError(
                f"num_keypoints={self.num_keypoints} exceeds the {height * width} pixels of a "
                f"{height}x{width} image; the detector cannot return that many keypoints."
            )
        features = self.keypoint_detector(images)
        keypoints_px = torch.stack([f.keypoints for f in features])
        scores = torch.stack([f.keypoint_scores for f in features])
        keypoints = normalize_pixel_coordinates(keypoints_px, height, width)
        descriptors = self.describe(
            images,
            keypoints,
            mode=mode,
            normalize=normalize,
            pad_if_not_divisible=pad_if_not_divisible,
        )
        return keypoints_px, scores, descriptors
