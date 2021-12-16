import inspect
from typing import Any, Optional


class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        self._children = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={self._module_dict})'
        return format_str

    def _register_module(
        self, module_class, module_name: Optional[str] = None, force: Optional[bool] = False
    ):
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}.')

        if module_name is None:
            module_name = module_class.__name__

        self._module_dict[module_name] = module_class
        setattr(self, module_name, module_class)

    def register_module(
        self, name: Optional[str] = None, force: Optional[bool] = False, module: Any = None
    ):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
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
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register
