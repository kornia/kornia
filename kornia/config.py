import os
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["kornia_config", "InstallationMode"]


class InstallationMode(str, Enum):
    # Ask the user if to install the dependencies
    ASK = "ASK"
    # Install the dependencies
    AUTO = "AUTO"
    # Raise an error if the dependencies are not installed
    RAISE = "RAISE"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()  # Case-insensitive comparison
        return super().__eq__(other)


class LazyLoaderConfig:
    _installation_mode: InstallationMode = InstallationMode.ASK

    @property
    def installation_mode(self) -> InstallationMode:
        return self._installation_mode

    @installation_mode.setter
    def installation_mode(self, value: str) -> None:
        # Allow setting via string by converting to the Enum
        if isinstance(value, str):
            try:
                self._installation_mode = InstallationMode(value.upper())
            except ValueError:
                raise ValueError(f"{value} is not a valid InstallationMode. Choose from: {list(InstallationMode)}")
        elif isinstance(value, InstallationMode):
            self._installation_mode = value
        else:
            raise TypeError("installation_mode must be a string or InstallationMode Enum.")


@dataclass
class KorniaConfig:
    hub_models_dir: str
    hub_onnx_dir: str
    output_dir: str = "kornia_outputs"
    hub_cache_dir: str = ".kornia_hub"
    lazyloader: LazyLoaderConfig = field(default_factory=LazyLoaderConfig)


kornia_config = KorniaConfig(
    hub_models_dir=os.path.join(".kornia_hub", "models"), hub_onnx_dir=os.path.join(".kornia_hub", "onnx_models")
)
