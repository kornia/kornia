import importlib
import logging
import subprocess
import sys
from types import ModuleType
from typing import List, Optional

from kornia.config import InstallationMode, kornia_config

logger = logging.getLogger(__name__)


class LazyLoader:
    """A class that implements lazy loading for Python modules.

    This class defers the import of a module until an attribute of the module is accessed.
    It helps in reducing the initial load time and memory usage of a script, especially when
    dealing with large or optional dependencies that might not be used in every execution.

    Attributes:
        module_name: The name of the module to be lazily loaded.
        module: The actual module object, initialized to None and loaded upon first access.
    """

    auto_install: bool = False

    def __init__(self, module_name: str, dev_dependency: bool = False) -> None:
        """Initializes the LazyLoader with the name of the module.

        Args:
            module_name: The name of the module to be lazily loaded.
            dev_dependency: If the dependency is required in the dev environment.
                If True, the module will be loaded in the dev environment.
                If False, the module will not be loaded in the dev environment.
        """
        self.module_name = module_name
        self.module: Optional[ModuleType] = None
        self.dev_dependency = dev_dependency

    def _install_package(self, module_name: str) -> None:
        logger.info(f"Installing `{module_name}` ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", module_name], shell=False, check=False)  # noqa: S603

    def _load(self) -> None:
        """Loads the module if it hasn't been loaded yet.

        This method is called internally when an attribute of the module is accessed for the first time. It attempts to
        import the module and raises an ImportError with a custom message if the module is not installed.
        """
        if not self.dev_dependency:
            if "--doctest-modules" in sys.argv:
                logger.info(f"Doctest detected, skipping loading of '{self.module_name}'")
                return
            try:
                if __sphinx_build__:  # type:ignore
                    logger.info(f"Sphinx detected, skipping loading of '{self.module_name}'")
                    return
            except NameError:
                pass

        if self.module is None:
            try:
                self.module = importlib.import_module(self.module_name)
            except ImportError as e:
                if kornia_config.lazyloader.installation_mode == InstallationMode.AUTO or self.auto_install:
                    self._install_package(self.module_name)
                elif kornia_config.lazyloader.installation_mode == InstallationMode.ASK:
                    to_ask = True
                    if_install = input(
                        f"Optional dependency '{self.module_name}' is not installed. "
                        "You may silent this prompt by `kornia_config.lazyloader.installation_mode = 'auto'`. "
                        "Do you wish to install the dependency? [Y]es, [N]o, [A]ll."
                    )
                    while to_ask:
                        if if_install.lower() == "y" or if_install.lower() == "yes":
                            self._install_package(self.module_name)
                            self.module = importlib.import_module(self.module_name)
                            to_ask = False
                        elif if_install.lower() == "a" or if_install.lower() == "all":
                            self.auto_install = True
                            self._install_package(self.module_name)
                            self.module = importlib.import_module(self.module_name)
                            to_ask = False
                        elif if_install.lower() == "n" or if_install.lower() == "no":
                            raise ImportError(
                                f"Optional dependency '{self.module_name}' is not installed. "
                                f"Please install it to use this functionality."
                            ) from e
                        else:
                            if_install = input("Invalid input. Please enter 'Y', 'N', or 'A'.")

                elif kornia_config.lazyloader.installation_mode == InstallationMode.RAISE:
                    raise ImportError(
                        f"Optional dependency '{self.module_name}' is not installed. "
                        f"Please install it to use this functionality."
                    ) from e
                self.module = importlib.import_module(self.module_name)

    def __getattr__(self, item: str) -> object:
        """Loads the module (if not already loaded) and returns the requested attribute.

        This method is called when an attribute of the LazyLoader instance is accessed.
        It ensures that the module is loaded and then returns the requested attribute.

        Args:
            item: The name of the attribute to be accessed.

        Returns:
            The requested attribute of the loaded module.
        """
        self._load()
        return getattr(self.module, item)

    def __dir__(self) -> List[str]:
        """Loads the module (if not already loaded) and returns the list of attributes of the module.

        This method is called when the built-in dir() function is used on the LazyLoader instance.
        It ensures that the module is loaded and then returns the list of attributes of the module.

        Returns:
            list: The list of attributes of the loaded module.
        """
        self._load()
        return dir(self.module)


# NOTE: This section is used for lazy loading of external modules. However, sphinx
#       would also try to support lazy loading of external modules. To avoid that, we
#       may set the module name to `autodoc_mock_imports` in conf.py to avoid undesired
#       installation of external modules.
numpy = LazyLoader("numpy", dev_dependency=True)
PILImage = LazyLoader("PIL.Image", dev_dependency=True)
onnx = LazyLoader("onnx", dev_dependency=True)
diffusers = LazyLoader("diffusers")
transformers = LazyLoader("transformers")
onnxruntime = LazyLoader("onnxruntime")
boxmot = LazyLoader("boxmot")
segmentation_models_pytorch = LazyLoader("segmentation_models_pytorch")
basicsr = LazyLoader("basicsr")
requests = LazyLoader("requests")
ivy = LazyLoader("ivy")
