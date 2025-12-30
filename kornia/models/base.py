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

from kornia.core import Module, Tensor
from kornia.core.external import PILImage as Image
from kornia.io import write_image
from kornia.utils.image import tensor_to_image

logger = logging.getLogger(__name__)

ModelConfig = TypeVar("ModelConfig")


class ModelBaseMixin:
    name: str = "model"

    def _tensor_to_type(
        self, output: Union[Tensor, List[Tensor]], output_type: str, is_batch: bool = False
    ) -> Union[Tensor, List[Tensor], List[Image.Image]]:  # type: ignore
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

    def save(self, output: Union[Tensor, List[Tensor]], directory: str, is_batch: bool = False) -> None:
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
                write_image(out, os.path.join(directory, f"{self.name}_{timestamp}_{i}.png"))
        else:
            write_image(output, os.path.join(directory, f"{self.name}_{timestamp}.png"))
        logger.info(f"Outputs are saved in {directory}")


class ModelBase(ABC, Module, Generic[ModelConfig]):
    """Abstract model class with some utilities function."""

    def load_checkpoint(self, checkpoint: str, device: Optional[torch.device] = None) -> None:
        """Load checkpoint from a given url or file.

        Args:
            checkpoint: The url or filepath for the respective checkpoint
            device: The desired device to load the weights and move the model

        """
        if os.path.isfile(checkpoint):
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
        else:
            state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location=device)

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
        compiled = torch.compile(
            self, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable
        )
        compiled = cast(ModelBase[ModelConfig], compiled)
        return compiled
