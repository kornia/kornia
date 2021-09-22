from enum import Enum


class TrainerState(Enum):
    TRAINING = 0
    TERMINATE = 1


class Configuration:
    def __init__(self, **entries):
        for k, v in entries.items():
            self.__dict__[k] = Configuration(**v) if isinstance(v, dict) else v

    @classmethod
    def from_yaml(cls, config_file: str):
        """Find the config / yml file and load the dataset metadata."""
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return cls(**data)
