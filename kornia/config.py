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

import os
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["InstallationMode", "kornia_config"]


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
                raise ValueError(
                    f"{value} is not a valid InstallationMode. Choose from: {list(InstallationMode)}"
                ) from None
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
