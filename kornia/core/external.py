import importlib
from types import ModuleType
from typing import List, Optional


class LazyLoader:
    """A class that implements lazy loading for Python modules.

    This class defers the import of a module until an attribute of the module is accessed.
    It helps in reducing the initial load time and memory usage of a script, especially when
    dealing with large or optional dependencies that might not be used in every execution.

    Attributes:
        module_name: The name of the module to be lazily loaded.
        module: The actual module object, initialized to None and loaded upon first access.
    """

    def __init__(self, module_name: str) -> None:
        """Initializes the LazyLoader with the name of the module.

        Args:
            module_name (str): The name of the module to be lazily loaded.
        """
        self.module_name = module_name
        self.module: Optional[ModuleType] = None

    def _load(self) -> None:
        """Loads the module if it hasn't been loaded yet.

        This method is called internally when an attribute of the module is accessed for the first time. It attempts to
        import the module and raises an ImportError with a custom message if the module is not installed.
        """
        if self.module is None:
            try:
                self.module = importlib.import_module(self.module_name)
            except ImportError as e:
                raise ImportError(
                    f"Optional dependency '{self.module_name}' is not installed. "
                    f"Please install it to use this functionality."
                ) from e

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


numpy = LazyLoader("numpy")
PILImage = LazyLoader("PIL.Image")
