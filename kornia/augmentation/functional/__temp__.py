import warnings


def __deprecation_warning(name: str, replacement: str):
    warnings.warn(f"`{name}` is no longer maintained and will be removed from the future versions. "
                  f"Please use {replacement} instead.", category=DeprecationWarning)
