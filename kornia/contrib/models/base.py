from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, cast

import torch

from kornia.core import Module

ModelConfig = TypeVar("ModelConfig")


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
        """This function should build/load the model.

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
        options: dict[Any, Any] = {},
        disable: bool = False,
    ) -> ModelBase[ModelConfig]:
        compiled = torch.compile(
            self, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable
        )
        compiled = cast(ModelBase[ModelConfig], compiled)
        return compiled
