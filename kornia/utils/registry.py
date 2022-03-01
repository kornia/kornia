import inspect
import re
import sys
from typing import Any, Callable, Dict, List, Optional


class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, object] = dict()
        # self._children = dict()  # TODO: enable sub-namespaces.

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self._module_dict.get(key) is not None

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + f'(name={self._name}, items={self._module_dict})'
        return format_str

    def list_modules(self, query: Optional[str] = None) -> List[str]:
        if query is not None:
            # Search for any modules by a regex string.
            raise NotImplementedError("Please contribute to Kornia if you are interested in this features.")
        return list(self._module_dict.keys())

    def _register_module(
        self, module_class: Any, module_name: Optional[str] = None, force: Optional[bool] = False
    ) -> None:
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}.')

        if module_name is None:
            module_name = module_class.__name__

        if not force and module_name in self._module_dict:
            raise RuntimeError(f"Module `{module_name}` existed. Found {self._module_dict.get(module_name)}.")

        self._module_dict[module_name] = module_class
        setattr(self, module_name, module_class)

    def register_module(
        self, name: Optional[str] = None, force: bool = False, module: Any = None
    ) -> Callable:
        """Register a module.

        A record will be added to `self._module_dict` and as a class attribute,
        whose key is the class name or the specified name, and value is the class itself.
        This function can be used as a decorator or a normal function.

        Args:
            name: The module name to be registered. If not specified, the class name will be used.
            force: Whether to override an existing class with the same name. Default: False.
            module: Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        # TODO: handle "xx.yy.zz" names with submodules.
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f' of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls: Any) -> Any:
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def register_modules_from_namespace(
        self, namespace: str, allowed_classes: Optional[list] = None,
        exclude_patterns: List[str] = []
    ) -> None:
        """Register modules under a namespace.

        Args:
            namespace: The namespace contains all needed modules.
            allowed_classes: restrict  are allowed. Default: None.
            exclude_patterns: regex patterns for class names to exclude particular modules.
        """
        def _predicate(module: Any) -> bool:
            if allowed_classes is not None:
                return inspect.isclass(module) and issubclass(module, tuple(allowed_classes))
            return inspect.isclass(module)

        pattern = [re.compile(p) for p in exclude_patterns]

        def _pattern_matching(name: str) -> bool:
            for p in pattern:
                if p.match(name) is not None:
                    return True
            return False

        classes = inspect.getmembers(sys.modules[namespace], _predicate)
        [self.register_module(name=n, module=cls) for n, cls in classes if not _pattern_matching(n)]
