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
from enum import StrEnum

__all__ = ["InstallationMode", "kornia_config"]


_INSTALLATION_MODE_ENV_VAR = "KORNIA_INSTALLATION_MODE"


class InstallationMode(StrEnum):
    """How :class:`kornia.core.external.LazyLoader` handles a missing optional dependency.

    Set the mode with ``kornia_config.lazyloader.installation_mode = "<mode>"`` or, before kornia is imported, with
    the ``KORNIA_INSTALLATION_MODE`` environment variable. Both accept ``"raise"``, ``"ask"`` and ``"auto"`` in any
    case. Members compare equal to their upper-case values (``InstallationMode.RAISE == "RAISE"``).

    - ``RAISE`` (the default): raise an ``ImportError`` that names the kornia extra to install, for example
      ``pip install "kornia[onnx]"``.
    - ``ASK``: on an interactive terminal, ask whether to install that extra. When stdin is not a terminal (a CI job
      or a DataLoader worker, for example), behave as ``RAISE``.
    - ``AUTO``: run ``pip install "kornia[<extra>]"`` for the declared kornia extra (instead of the import name) with
      the running interpreter, and raise an ``ImportError`` if pip fails.

    A dependency that declares no kornia extra is never installed: ``ASK`` and ``AUTO`` raise the same
    ``ImportError`` as ``RAISE`` for it.
    """

    # Ask on an interactive terminal whether to install the extra; raise otherwise
    ASK = "ASK"
    # Install the declared kornia extra without asking
    AUTO = "AUTO"
    # Raise an ImportError naming the kornia extra to install
    RAISE = "RAISE"


def _parse_installation_mode(value: object, source: str) -> InstallationMode:
    """Return the :class:`InstallationMode` a case-insensitive string names.

    Args:
        value: the value to parse, for example ``"raise"`` or ``InstallationMode.AUTO``.
        source: the name the value was given under, used in the error message.

    Raises:
        TypeError: if ``value`` is not a string.
        ValueError: if ``value`` names no mode.

    """
    if not isinstance(value, str):
        raise TypeError("installation_mode must be a string or InstallationMode Enum.")
    try:
        return InstallationMode(value.upper())
    except ValueError:
        choices = ", ".join(repr(mode.value.lower()) for mode in InstallationMode)
        raise ValueError(
            f"{source}={value!r} is not a valid installation mode. Choose from: {choices} (any case)."
        ) from None


class LazyLoaderConfig:
    """Configure lazy loading behavior for external dependencies.

    The initial ``installation_mode`` is read from the ``KORNIA_INSTALLATION_MODE`` environment variable when it is
    set and not empty, and is :attr:`InstallationMode.RAISE` otherwise.
    """

    def __init__(self) -> None:
        env_value = os.environ.get(_INSTALLATION_MODE_ENV_VAR, "")
        self._installation_mode = (
            _parse_installation_mode(env_value, _INSTALLATION_MODE_ENV_VAR)
            if env_value.strip()
            else InstallationMode.RAISE
        )

    @property
    def installation_mode(self) -> InstallationMode:
        """How a missing optional dependency is handled; see :class:`InstallationMode`."""
        return self._installation_mode

    @installation_mode.setter
    def installation_mode(self, value: str) -> None:
        self._installation_mode = _parse_installation_mode(value, "installation_mode")


@dataclass
class KorniaConfig:
    """Configure Kornia's behavior."""

    hub_models_dir: str
    hub_onnx_dir: str
    output_dir: str = "kornia_outputs"
    hub_cache_dir: str = ".kornia_hub"
    lazyloader: LazyLoaderConfig = field(default_factory=LazyLoaderConfig)


kornia_config = KorniaConfig(
    hub_models_dir=os.path.join(".kornia_hub", "models"), hub_onnx_dir=os.path.join(".kornia_hub", "onnx_models")
)
