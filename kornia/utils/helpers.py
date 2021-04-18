from typing import Tuple, List, Any, Optional

import torch


def _extract_device_dtype(tensor_list: List[Optional[Any]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are torch.Tensor).

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                continue
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


def _torch_inverse_cast(input: torch.Tensor) -> torch.Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does, is cast
    input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    assert isinstance(input, torch.Tensor), f"Input must be torch.Tensor. Got: {type(input)}."
    dtype: torch.dtype = input.dtype
    if dtype not in (torch.float32, torch.float64,):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)
