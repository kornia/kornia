import warnings
from functools import wraps
from typing import Callable

warnings.simplefilter('always', DeprecationWarning)


def __deprecation_warning(name: str, replacement: str):
    warnings.warn(
        f"`{name}` will be renamed to `{replacement}` in the future versions. " f"Please use `{replacement}` instead.",
        category=DeprecationWarning,
    )


def _deprecation_wrapper(new_func: Callable, old_func: str) -> Callable:
    @wraps(new_func)
    def wrapper(*args, **kwargs):
        __deprecation_warning(old_func, new_func.__name__)
        return new_func(*args, **kwargs)

    return wrapper
