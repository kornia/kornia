from typing import Tuple, Union, List, cast, Optional

import torch

from kornia.constants import Resample


def _extract_device_dtype(tensor_list: List[Optional[torch.Tensor]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input tensors are in the same device.

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                raise ValueError(f"Expected None or Tensor. Got {tensor}.")
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError("Passed values are not in the same device and dtype."
                                 f"Got ({device}, {dtype}) and ({_device}, {_dtype}).")
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    return (device, dtype)


def _parse_align_corners(align_corners: Optional[bool], resample: Resample) -> Optional[bool]:
    r"""Set a sensible default value for ``align_corners`` used in :func:`torch.nn.functional.interpolate`.

    ``align_corners`` has to be ``False`` for the interpolation modes ``"linear"``, ``"bilinear"``, ``"bicubic"``, and
    ``"trilinear"`` to suppress a warning. For all other interpolation modes ``align_corners`` is not a valid parameter
    and has to be ``None``.

    Args:
        align_corners: If not ``None``, i.e. the default value, it is returned as is.
        resample: Interpolation mode.
    """
    if align_corners is not None:
        return align_corners

    return (
        False
        if resample.name.lower() in ("linear", "bilinear", "bicubic", "trilinear")
        else None
    )
