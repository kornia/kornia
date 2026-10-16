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

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kornia.core import ImageSequential
from kornia.enhance.normalize import Normalize
from kornia.enhance.rescale import Rescale
from kornia.geometry.transform import Resize, resize

# PIL resampling filters of a processor config's ``resample`` that kornia's resize implements
_RESAMPLE_TO_INTERPOLATION = {0: "nearest", 2: "bilinear", 3: "bicubic"}


def _dpt_output_size(
    height: int, width: int, size: tuple[int, int], keep_aspect_ratio: bool, multiple: int
) -> tuple[int, int]:
    """Return the ``(height, width)`` to which transformers' ``DPTImageProcessor`` resizes an image.

    A transcription of ``get_resize_output_image_size`` and its ``constrain_to_multiple_of`` in
    ``transformers.models.dpt.image_processing_dpt``. With ``keep_aspect_ratio``, the side whose scale to ``size`` is
    closer to 1 is fitted to ``size`` and the other side is scaled by the same factor; without it, each side is scaled
    to its target. Each side is then rounded to the nearest multiple of ``multiple`` with Python ``round``, so a tie
    goes to the even multiple.
    """
    scale_height = size[0] / height
    scale_width = size[1] / width
    if keep_aspect_ratio:
        # scale as little as possible
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width
        else:
            scale_width = scale_height
    return round(scale_height * height / multiple) * multiple, round(scale_width * width / multiple) * multiple


class _DPTResize(nn.Module):
    """Resize an image as transformers' ``DPTImageProcessor`` does, to a size computed from each input's shape.

    The output size is :func:`_dpt_output_size` of the input's height and width. ``"bilinear"`` and ``"bicubic"``
    resize with ``align_corners=False`` and antialiasing; ``"nearest"`` takes the nearest pixel without filtering, as
    PIL's ``NEAREST`` does. The size depends on the input shape, so this module runs in torch: it cannot be exported to
    an ONNX graph with dynamic height and width.

    Args:
        size: Target ``(height, width)`` before the aspect ratio and the multiple are applied.
        keep_aspect_ratio: Whether to scale both sides by the same factor.
        multiple: Each output side is a multiple of this number.
        interpolation: ``"bicubic"``, ``"bilinear"`` or ``"nearest"``.
    """

    def __init__(self, size: tuple[int, int], keep_aspect_ratio: bool, multiple: int, interpolation: str) -> None:
        super().__init__()
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.multiple = multiple
        self.interpolation = interpolation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Resize ``image`` of shape :math:`(*, C, H, W)` to the DPT output size of its ``H`` and ``W``."""
        height, width = image.shape[-2:]
        output_size = _dpt_output_size(height, width, self.size, self.keep_aspect_ratio, self.multiple)
        if self.interpolation == "nearest":
            return resize(image, output_size, interpolation="nearest")
        return resize(image, output_size, interpolation=self.interpolation, align_corners=False, antialias=True)


class PreprocessingLoader:
    """Factory for converting image-processor settings into Kornia modules.

    Hugging Face image processor configuration files describe preprocessing as JSON
    fields, for example whether to resize an image, multiply it by a rescale factor,
    or normalize it with channel statistics. This helper maps the supported fields to
    Kornia modules so the preprocessing can be composed with ONNX model pipelines.
    """

    @staticmethod
    def normalize(mean: torch.Tensor, std: torch.Tensor) -> Normalize:
        """Create a channel-wise normalization module.

        Args:
            mean: Mean value used for each channel. The tensor is passed directly to
                :class:`~kornia.enhance.normalize.Normalize`.
            std: Standard deviation value used for each channel.

        Returns:
            Kornia normalization module that subtracts ``mean`` and divides by ``std``
            for every image in the batch.
        """
        return Normalize(mean=mean, std=std)

    @staticmethod
    def rescale(rescale_factor: float) -> Rescale:
        """Create an intensity rescaling module.

        Args:
            rescale_factor: Multiplicative factor applied to image values. In the DPT
                pipeline this is typically derived from the processor JSON so the
                tensor range matches what the exported model expects.

        Returns:
            Kornia rescaling module that multiplies the input image tensor by
            ``rescale_factor``.
        """
        return Rescale(factor=rescale_factor)

    @staticmethod
    def resize(width: int, height: int) -> Resize:
        """Create an image resize module from width and height values.

        Args:
            width: Target image width, corresponding to the last spatial dimension
                ``W`` of an image tensor.
            height: Target image height, corresponding to the second-to-last spatial
                dimension ``H`` of an image tensor.

        Returns:
            Kornia resize module configured with output size ``(height, width)``.
        """
        return Resize((height, width))

    @staticmethod
    def from_json(req: dict[str, Any]) -> ImageSequential:
        """Build a preprocessing pipeline from a processor configuration dictionary.

        Args:
            req: Parsed processor configuration. The dictionary must contain
                ``image_processor_type`` so the loader can choose the matching
                processor-specific parser.

        Returns:
            Sequential Kornia module containing the supported preprocessing steps.

        Raises:
            RuntimeError: If the requested image processor type is not supported by
                this loader.
        """
        if req["image_processor_type"] == "DPTImageProcessor":
            return DPTImageProcessor.from_json(req)
        raise RuntimeError(f"Unsupported image processor type: {req['image_processor_type']}")


class DPTImageProcessor(PreprocessingLoader):
    """Parser for DPT image processor configurations.

    This parser reads the fields of a transformers ``DPTImageProcessor`` configuration and builds a Kornia pipeline
    in the processor's order: resize, rescale, and normalize. The resize follows the processor's rule: it keeps the
    aspect ratio when ``keep_aspect_ratio`` is set and makes each side a multiple of ``ensure_multiple_of``, so its
    output size depends on the input's. That resize runs in torch; it cannot be part of an ONNX graph with dynamic
    height and width.

    Padding (``do_pad`` with its ``size_divisor``) is not supported, nor are the ``resample`` filters other than
    nearest, bilinear and bicubic.
    """

    @staticmethod
    def from_json(json_data: dict[str, Any]) -> ImageSequential:
        """Convert a DPT processor JSON dictionary into a Kornia preprocessing pipeline.

        Args:
            json_data: Parsed DPT processor configuration. The fields read are
                ``do_pad``, which must be false, ``do_resize``, ``size``, ``keep_aspect_ratio``,
                ``ensure_multiple_of``, ``resample``, ``do_rescale``, ``rescale_factor``,
                ``do_normalize``, ``image_mean`` and ``image_std``. An absent ``keep_aspect_ratio``,
                ``ensure_multiple_of`` or ``resample`` takes the processor's default:
                ``False``, ``1`` and ``3`` (bicubic).

        Returns:
            :class:`~kornia.core.ImageSequential` containing the enabled preprocessing
            modules in execution order.

        Raises:
            NotImplementedError: If the configuration requests padding, or a ``resample``
                filter other than nearest (0), bilinear (2) or bicubic (3).
        """
        preproc_list: list[nn.Module] = []
        if json_data["do_pad"]:
            raise NotImplementedError
        if json_data["do_resize"]:
            resample = json_data.get("resample", 3)
            if resample not in _RESAMPLE_TO_INTERPOLATION:
                raise NotImplementedError(
                    f"Unsupported resample {resample}: kornia implements nearest (0), bilinear (2) and bicubic (3)."
                )
            preproc_list.append(
                _DPTResize(
                    size=(json_data["size"]["height"], json_data["size"]["width"]),
                    keep_aspect_ratio=json_data.get("keep_aspect_ratio", False),
                    multiple=json_data.get("ensure_multiple_of", 1),
                    interpolation=_RESAMPLE_TO_INTERPOLATION[resample],
                )
            )
        if json_data["do_rescale"]:
            preproc_list.append(PreprocessingLoader.rescale(rescale_factor=json_data["rescale_factor"] * 255))
        if json_data["do_normalize"]:
            preproc_list.append(
                PreprocessingLoader.normalize(
                    mean=torch.tensor([json_data["image_mean"]]), std=torch.tensor([json_data["image_std"]])
                )
            )
        return ImageSequential(*preproc_list)
