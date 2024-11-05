from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Tuple

from kornia.core import Module, Tensor
from kornia.metrics.average_meter import AverageMeter

# import yaml


class TrainerState(Enum):
    STARTING = 0
    TRAINING = 1
    VALIDATE = 2
    TERMINATE = 3


# NOTE: this class needs to be redefined according to the needed parameters.
@dataclass
class Configuration:
    data_path: str = field(default="./", metadata={"help": "The input data directory."})
    batch_size: int = field(default=1, metadata={"help": "The number of batches for the training dataloader."})
    num_epochs: int = field(default=1, metadata={"help": "The number of epochs to run the training."})
    lr: float = field(default=1e-3, metadata={"help": "The learning rate to be used for the optimize."})
    output_path: str = field(default="./output", metadata={"help": "The output data directory."})
    image_size: Tuple[int, int] = field(default=(224, 224), metadata={"help": "The input image size."})

    # TODO: possibly remove because hydra already do this
    # def __init__(self, **entries):
    #     for k, v in entries.items():
    #         self.__dict__[k] = Configuration(**v) if isinstance(v, dict) else v

    # @classmethod
    # def from_yaml(cls, config_file: str):
    #     """Create an instance of the configuration from a yaml file."""
    #     with open(config_file) as f:
    #         data = yaml.safe_load(f)
    #     return cls(**data)


class Lambda(Module):
    """Module to create a lambda function as Module.

    Args:
        fcn: a pointer to any function.

    Example:
        >>> import torch
        >>> import kornia as K
        >>> fcn = Lambda(lambda x: K.geometry.resize(x, (32, 16)))
        >>> fcn(torch.rand(1, 4, 64, 32)).shape
        torch.Size([1, 4, 32, 16])
    """

    def __init__(self, fcn: Callable[..., Any]) -> None:
        super().__init__()
        self.fcn = fcn

    def forward(self, x: Tensor) -> Any:
        return self.fcn(x)


class StatsTracker:
    """Stats tracker for computing metrics on the fly."""

    def __init__(self) -> None:
        self._stats: Dict[str, AverageMeter] = {}

    @property
    def stats(self) -> Dict[str, AverageMeter]:
        return self._stats

    def update(self, key: str, val: float, batch_size: int) -> None:
        """Update the stats by the key value pair."""
        if key not in self._stats:
            self._stats[key] = AverageMeter()
        self._stats[key].update(val, batch_size)

    def update_from_dict(self, dic: Dict[str, float], batch_size: int) -> None:
        """Update the stats by the dict."""
        for k, v in dic.items():
            self.update(k, v, batch_size)

    def __repr__(self) -> str:
        return " ".join([f"{k.upper()}: {v.val:.2f} {v.val:.2f} " for k, v in self._stats.items()])

    def as_dict(self) -> Dict[str, AverageMeter]:
        """Return the dict format."""
        return self._stats
