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
from kornia.geometry.transform import Resize


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

    DPT models use a fixed image preprocessing recipe before inference. This parser
    reads the supported JSON fields and constructs the equivalent Kornia pipeline in
    the same order used by the processor configuration: resize, rescale, and normalize.
    Padding is intentionally not handled here because the current implementation does
    not include the extra parameters needed to reproduce that branch safely.
    """

    @staticmethod
    def from_json(json_data: dict[str, Any]) -> ImageSequential:
        """Convert a DPT processor JSON dictionary into a Kornia preprocessing pipeline.

        Args:
            json_data: Parsed DPT processor configuration. Supported fields include
                ``do_resize``, ``size``, ``do_rescale``, ``rescale_factor``,
                ``do_normalize``, ``image_mean``, and ``image_std``.

        Returns:
            :class:`~kornia.core.ImageSequential` containing the enabled preprocessing
            modules in execution order.

        Raises:
            NotImplementedError: If the configuration requests padding, which is not
                implemented by this loader.
        """
        preproc_list: list[nn.Module] = []
        if json_data["do_pad"]:
            raise NotImplementedError
        if json_data["do_resize"]:
            # Missing some parameters such as `ensure_multiple_of`, `keep_aspect_ratio`
            preproc_list.append(
                PreprocessingLoader.resize(width=json_data["size"]["width"], height=json_data["size"]["height"])
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
