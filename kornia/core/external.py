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

import builtins
import importlib
import logging
import subprocess
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

from kornia.config import InstallationMode, kornia_config

logger = logging.getLogger(__name__)

# The loader's own instance attributes. ``__getattr__`` sees them only on a copy or an unpickled loader whose state is
# not restored yet; answering them from the module would recurse.
_LOADER_ATTRIBUTES = frozenset({"module_name", "module", "dev_dependency", "extra", "_install_error"})


def _stdin_is_a_terminal() -> bool:
    """Return whether ``input()`` can reach a person: stdin exists, is open and is a TTY."""
    stdin = sys.stdin
    if stdin is None:
        return False
    try:
        return stdin.isatty()
    except (AttributeError, OSError, ValueError):  # replaced by an object without isatty, or closed
        return False


class LazyLoader:
    """A class that implements lazy loading for Python modules.

    This class defers the import of a module until an attribute of the module is accessed.
    It helps in reducing the initial load time and memory usage of a script, especially when
    dealing with large or optional dependencies that might not be used in every execution.

    What happens when the module is missing is set by ``kornia.config.kornia_config.lazyloader.installation_mode``
    (or the ``KORNIA_INSTALLATION_MODE`` environment variable); see :class:`kornia.config.InstallationMode`. By
    default an ``ImportError`` names the kornia extra to install.

    Attributes:
        module_name: The name of the module to be lazily loaded.
        module: The actual module object, initialized to None and loaded upon first access.
        extra: The name of the kornia optional-dependency extra that provides the module, if any.

    """

    def __init__(self, module_name: str, dev_dependency: bool = False, extra: Optional[str] = None) -> None:
        """Initialize the LazyLoader with the name of the module.

        Args:
            module_name: The name of the module to be lazily loaded.
            dev_dependency: Whether kornia's documentation build needs the module. If False, the Sphinx build of
                kornia's documentation (``docs/source/conf.py`` sets a ``__sphinx_build__`` builtin) does not import
                it: the loader stays empty there and attribute access raises ``AttributeError``. It has no effect
                elsewhere.
            extra: The name of the kornia optional-dependency extra that installs the module, e.g. ``"onnx"``.
                When set, the "not installed" messages tell the user to run ``pip install "kornia[<extra>]"``, and
                the ``"ask"`` and ``"auto"`` installation modes install that extra. Without it, nothing is installed.

        """
        self.module_name = module_name
        self.module: Optional[ModuleType] = None
        self.dev_dependency = dev_dependency
        self.extra = extra
        self._install_error: Optional[str] = None

    @property
    def _install_hint(self) -> str:
        """Return the trailing sentence of the "not installed" messages."""
        if self.extra is not None:
            return f'Install it with: pip install "kornia[{self.extra}]".'
        return "Please install it to use this functionality."

    def _should_install(self) -> bool:
        """Decide, from the installation mode, whether to install the declared extra of a missing module."""
        if self.extra is None:
            return False
        mode = kornia_config.lazyloader.installation_mode
        if mode == InstallationMode.AUTO:
            return True
        if mode == InstallationMode.ASK:
            return self._ask_to_install()
        return False

    def _ask_to_install(self) -> bool:
        """Ask on the terminal whether to install the declared extra; ``False`` when stdin is not a terminal."""
        if not _stdin_is_a_terminal():
            return False
        question = (
            f"Optional dependency '{self.module_name}' is not installed. "
            f'Install it now with `pip install "kornia[{self.extra}]"`? '
            "[Y]es, [N]o, [A]ll (install every missing kornia extra without asking for the rest of this session). "
            "Set `kornia_config.lazyloader.installation_mode` or the KORNIA_INSTALLATION_MODE environment variable "
            "to 'raise' to never be asked, or to 'auto' to always install. "
        )
        try:
            answer = input(question)
            while True:
                choice = answer.strip().lower()
                if choice in ("y", "yes"):
                    return True
                if choice in ("n", "no"):
                    return False
                if choice in ("a", "all"):
                    kornia_config.lazyloader.installation_mode = InstallationMode.AUTO
                    return True
                answer = input("Please answer 'y', 'n' or 'a'. ")
        except EOFError:
            return False

    def _install_extra(self) -> None:
        """Install the declared extra with this interpreter's pip and import the module.

        Raises:
            ImportError: if pip fails or the module still cannot be imported. The failure is remembered, so later
                accesses raise it again without running pip again.

        """
        requirement = f"kornia[{self.extra}]"
        command = [sys.executable, "-m", "pip", "install", requirement]
        logger.info("Installing %s for the optional dependency '%s' ...", requirement, self.module_name)
        try:
            subprocess.run(command, check=True)  # noqa: S603
        except (OSError, subprocess.CalledProcessError) as e:
            self._install_error = (
                f"Optional dependency '{self.module_name}' is not installed, and "
                f'`pip install "{requirement}"` failed: {e}'
            )
            raise ImportError(self._install_error) from e
        importlib.invalidate_caches()
        try:
            self.module = importlib.import_module(self.module_name)
        except ImportError as e:
            self._install_error = (
                f"Optional dependency '{self.module_name}' cannot be imported after `pip install \"{requirement}\"`."
            )
            raise ImportError(self._install_error) from e

    def _load(self) -> None:
        """Import the module on first use.

        A missing module is handled according to ``kornia_config.lazyloader.installation_mode`` (see
        :class:`kornia.config.InstallationMode`): by default it raises an ImportError whose message names the kornia
        extra to install.

        Raises:
            ImportError: if the module is missing and is not installed.

        """
        if self.module is not None:
            return
        if not self.dev_dependency and getattr(builtins, "__sphinx_build__", False):
            logger.info(f"Sphinx detected, skipping loading of '{self.module_name}'")
            return
        try:
            self.module = importlib.import_module(self.module_name)
            return
        except ImportError as e:
            import_error = e
        if self._install_error is not None:
            raise ImportError(self._install_error) from import_error
        if not self._should_install():
            raise ImportError(
                f"Optional dependency '{self.module_name}' is not installed. {self._install_hint}"
            ) from import_error
        self._install_extra()

    def __getattr__(self, item: str) -> object:
        """Load the module (if not already loaded) and returns the requested attribute.

        This method is called when an attribute of the LazyLoader instance is accessed.
        It ensures that the module is loaded and then returns the requested attribute.

        Protocol lookups (dunder names such as ``__wrapped__`` or ``__deepcopy__``, which ``copy``, ``pickle``,
        ``inspect.unwrap`` and doctest collection make) never import, install or ask: before the module is loaded
        they raise ``AttributeError``.

        Args:
            item: The name of the attribute to be accessed.

        Returns:
            The requested attribute of the loaded module.

        """
        if item in _LOADER_ATTRIBUTES:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")
        if item.startswith("__") and item.endswith("__"):
            module = self.__dict__.get("module")
            if module is None:
                raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")
            return getattr(module, item)
        self._load()
        return getattr(self.module, item)

    def __getstate__(self) -> Dict[str, Any]:
        """Return the loader's state for copy and pickle, without the module object (the copy imports it again)."""
        state = self.__dict__.copy()
        state["module"] = None
        return state

    def __dir__(self) -> List[str]:
        """Load the module (if not already loaded) and returns the list of attributes of the module.

        This method is called when the built-in dir() function is used on the LazyLoader instance.
        It ensures that the module is loaded and then returns the list of attributes of the module.

        Returns:
            list: The list of attributes of the loaded module.

        """
        self._load()
        return dir(self.module)


# NOTE: kornia's Sphinx build (docs/source/conf.py sets a ``__sphinx_build__`` builtin) leaves the loaders created
#       with ``dev_dependency=False`` empty instead of importing their modules, so the documentation environment does
#       not need those packages.
numpy = LazyLoader("numpy", dev_dependency=True)
PILImage = LazyLoader("PIL.Image", dev_dependency=True, extra="image")
onnx = LazyLoader("onnx", dev_dependency=True, extra="onnx")
diffusers = LazyLoader("diffusers", extra="sd")
onnxruntime = LazyLoader("onnxruntime", extra="onnx")
