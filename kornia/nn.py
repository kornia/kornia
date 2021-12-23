# Added a leading score to avoid showing in the auto-completiong
import sys as _sys
import inspect as _inspect
import torch as _torch
from kornia.utils.registry import Registry as _Registry

_sys.modules["kornia.nn"] = _sys.modules[__name__]

_nn = _Registry("nn")

for _k, _v in _inspect.getmembers(_nn, _inspect.ismethod):
    if not _k.startswith("_"):
        setattr(_sys.modules[__name__], _k, _v)


def _register():
    """Lazy registering for resolving the package import ordering issue.
    """
    _nn.register_modules_from_namespace(
        "kornia.augmentation", allowed_classes=[_torch.nn.Module], exclude_patterns=[".*Base.?D"])
    _nn.register_modules_from_namespace("kornia.color", allowed_classes=[_torch.nn.Module])
    # Make every valid function accessible
    for k, v in _nn._module_dict.items():
        setattr(_sys.modules[__name__], k, v)
