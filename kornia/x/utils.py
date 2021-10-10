# TODO: remove the type: ignore in below after deprecating python 3.6
from dataclasses import dataclass, field  # type: ignore
from enum import Enum

import torch.nn as nn

# import yaml  # type: ignore


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
    image_size: tuple = field(default=(224, 224), metadata={"help": "The input image size."})

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


class Lambda(nn.Module):
    """Module to create a lambda function as nn.Module.

    Args:
        fcn: a pointer to any function.

    Example:
        >>> import torch
        >>> import kornia as K
        >>> fcn = Lambda(lambda x: K.geometry.resize(x, (32, 16)))
        >>> fcn(torch.rand(1, 4, 64, 32)).shape
        torch.Size([1, 4, 32, 16])
    """
    def __init__(self, fcn):
        super().__init__()
        self.fcn = fcn

    def forward(self, x):
        return self.fcn(x)
