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

import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, Union, cast

import torch
from torch import nn

from kornia.core.download import load_state_dict_from_url
from kornia.core.external import PILImage as Image
from kornia.image.utils import tensor_to_image
from kornia.io import write_image

logger = logging.getLogger(__name__)

ModelConfig = TypeVar("ModelConfig")


def _to_writable_png(image: torch.Tensor) -> torch.Tensor:
    """Convert a visualization tensor to the dtype ``write_image`` accepts for PNG.

    ``visualize`` returns float images in ``[0, 1]``, but ``write_image`` writes
    PNG only for ``uint8``/``uint16``; float32 is TIFF-only and every other
    float dtype is rejected outright.

    These are pictures for a human to look at, not data to round-trip, so the
    conversion is to ``uint8`` -- the same thing ``ImageModule`` already does
    before handing a tensor to PIL. Values are clamped first: a visualization
    that overshoots ``[0, 1]`` would otherwise wrap and put black where it
    should be white. Non-float images pass through untouched, so a ``uint8``
    or ``uint16`` visualization is written as-is.
    """
    if not image.is_floating_point():
        return image
    return (image.detach().clamp(0.0, 1.0) * 255).round().to(torch.uint8)


def _write_png_batch(path_stem: str, image: torch.Tensor) -> None:
    """Write one image, or one file per item of a batch.

    ``write_image`` takes ``(3, H, W)``, ``(1, H, W)`` or ``(H, W)``, but the
    containers document their inputs as ``(B, 3, H, W)`` and passed the batch
    straight through. The backend then rejects the rank with an error that
    names neither -- ``TypeError: argument 'image': 'ndarray' object is not an
    instance of 'ndarray'`` -- so a batch has to be split here.

    A batch of one still gets an index suffix, so a caller does not have to
    guess whether a file will be ``name.png`` or ``name_0.png``.
    """
    image = _to_writable_png(image)
    if image.dim() == 4:
        for i, item in enumerate(image):
            write_image(f"{path_stem}_{i}.png", item)
    else:
        write_image(f"{path_stem}.png", image)


class ModelBaseMixin:
    """Provide common properties and utilities for Kornia model classes."""

    name: str = "model"

    def _tensor_to_type(
        self, output: Union[torch.Tensor, List[torch.Tensor]], output_type: str, is_batch: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], List[Image.Image]]:  # type: ignore
        """Convert the output tensor to the desired type.

        Args:
            output: The output tensor or list of tensors.
            output_type: The desired output type. Accepted values are "torch" and "pil".
            is_batch: If True, the output is expected to be a batch of tensors.

        Returns:
            The converted output tensor or list of tensors.

        Raises:
            RuntimeError: If the output type is not supported.

        """
        if output_type == "torch":
            return output
        elif output_type == "pil":
            if isinstance(output, list):
                return [tensor_to_image(t) for t in output]
            else:
                return tensor_to_image(output)
        else:
            raise RuntimeError(f"Output type {output_type} is not supported. Accepted values are 'torch' and 'pil'.")

    def save(self, output: Union[torch.Tensor, List[torch.Tensor]], directory: str, is_batch: bool = False) -> None:
        """Save the output tensor to a directory.

        Args:
            output: The output tensor or list of tensors.
            directory: The directory to save the output.
            is_batch: If True, the output is expected to be a batch of tensors.

        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        if isinstance(output, list):
            for i, out in enumerate(output):
                _write_png_batch(os.path.join(directory, f"{self.name}_{timestamp}_{i}"), out)
        else:
            _write_png_batch(os.path.join(directory, f"{self.name}_{timestamp}"), output)
        logger.info(f"Outputs are saved in {directory}")

    def _save_outputs(
        self, output: Union[torch.Tensor, List[torch.Tensor]], directory: Optional[str] = None, suffix: str = ""
    ) -> None:
        """Save the output tensor to a directory with an optional suffix.

        Args:
            output: The output tensor or list of tensors.
            directory: The directory to save the output. If None, a default directory is used.
            suffix: Optional suffix to add to the filename.

        """
        if directory is None:
            name = f"{self.name}{suffix}_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d%H%M%S')!s}"
            directory = os.path.join("kornia_outputs", name)

        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        if isinstance(output, list):
            for i, out in enumerate(output):
                _write_png_batch(os.path.join(directory, f"{self.name}{suffix}_{timestamp}_{i}"), out)
        else:
            _write_png_batch(os.path.join(directory, f"{self.name}{suffix}_{timestamp}"), output)
        logger.info(f"Outputs are saved in {directory}")


class ModelBase(ABC, nn.Module, ModelBaseMixin, Generic[ModelConfig]):
    """Abstract model class with some utilities function."""

    def load_checkpoint(self, checkpoint: str | list[str], device: Optional[torch.device] = None) -> None:
        """Load checkpoint from a given url or file.

        Args:
            checkpoint: The url or filepath for the respective checkpoint
            device: The desired device to load the weights and move the model

        """
        if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
        else:
            state_dict = load_state_dict_from_url(checkpoint, map_location=device)

        self.load_state_dict(state_dict)

    @staticmethod
    @abstractmethod
    def from_config(config: ModelConfig) -> ModelBase[ModelConfig]:
        """Build/load the model.

        Args:
            config: The specifications for the model be build/loaded

        """
        raise NotImplementedError

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: Optional[str] = None,
        options: Optional[dict[Any, Any]] = None,
        disable: bool = False,
    ) -> ModelBase[ModelConfig]:
        """Compile this model with :func:`torch.compile`.

        Args:
            fullgraph: Whether Dynamo should require a single full graph.
            dynamic: Whether dynamic shape tracing is enabled.
            backend: Compilation backend name passed to :func:`torch.compile`.
            mode: Optional backend-specific compilation mode.
            options: Optional backend-specific option dictionary.
            disable: If ``True``, return an uncompiled model wrapper according
                to PyTorch's compile semantics.

        Returns:
            Compiled model object with the same high-level interface as this
            instance.
        """
        compiled = torch.compile(
            self, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable
        )
        compiled = cast(ModelBase[ModelConfig], compiled)
        return compiled
