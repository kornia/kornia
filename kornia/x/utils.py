from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml  # type: ignore


class TrainerState(Enum):
    TRAINING = 0
    TERMINATE = 1


# NOTE: this class needs to be redefined according to the needed parameters.
@dataclass
class Configuration:
    data_path: str = field(default="./", metadata={"help": "The input data directory."})
    batch_size: int = field(default=1, metadata={"help": "The number of batches for the training dataloader."})
    num_epochs: int = field(default=1, metadata={"help": "The number of epochs to run the training."})
    lr: float = field(default=1e-3, metadata={"help": "The learning rate to be used for the optimize."})
    output_path: str = field(default="./output", metadata={"help": "The output data directory."})

    def __init__(self, **entries):
        for k, v in entries.items():
            self.__dict__[k] = Configuration(**v) if isinstance(v, dict) else v

    @classmethod
    def from_yaml(cls, config_file: str):
        """Create an instance of the configuration from a yaml file."""
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return cls(**data)
