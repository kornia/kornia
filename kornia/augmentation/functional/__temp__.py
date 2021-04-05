from typing import Callable

import warnings
from functools import wraps


def __deprecation_warning(name: str, replacement: str):
    warnings.warn(f"`{name}` is no longer maintained and will be removed from the future versions. "
                  f"Please use {replacement} instead.", category=DeprecationWarning)


def _deprecation_wrapper(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"`{f.__name__}` is no longer maintained and will be removed from the future versions. ",
            category=DeprecationWarning)
        return f(*args, **kwargs)
    return wrapper
