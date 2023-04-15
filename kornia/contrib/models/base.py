from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

import torch

from kornia.core import Module

ModelType = TypeVar('ModelType', bound=Enum)


class ModelBase(ABC, Module, Generic[ModelType]):
    """Abstract model class with some utilities function."""

    def load_checkpoint(self, checkpoint: str, device: torch.device | None = None) -> None:
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
    def build(model_type: str | int | ModelType) -> ModelBase[ModelType]:
        """This function should build the desired model type.

        Args:
            model_type: The mapping for the desired model type/size
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_pretrained(
        model_type: str | int | ModelType, checkpoint: str | None = None, device: torch.device | None = None
    ) -> ModelBase[ModelType]:
        """This function should build the desired model type, load the checkpoint and move to device.

        Args:
            model_type: The mapping for the desired model type/size
            checkpoint: A URL or a path for the weights/states to be loaded
            device: The target device to allocate the model.
        """
        raise NotImplementedError
