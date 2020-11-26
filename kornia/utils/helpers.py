from typing import Tuple, Union, List, cast, Optional

import torch


<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
<<<<<<< master
<<<<<<< refs/remotes/kornia/master
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
<<<<<<< master
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
def _extract_device_dtype(tensor_list: List[Optional[torch.Tensor]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input tensors are in the same device.

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
<<<<<<< master
>>>>>>> [Fix] MotionBlur bug fix and doctest update (#782)
=======
def _extract_device_dtype(tensor_list: List[Optional[torch.Tensor]]):
    """This function will check if all the input tensors are in the same device.

    If so, it would return a tuple of (device, dtype)
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
=======
def _extract_device_dtype(tensor_list: List[Optional[torch.Tensor]]) -> Tuple[torch.device, torch.dtype]:
    """Check if all the input tensors are in the same device.

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> [Fix] MotionBlur bug fix and doctest update (#782)
=======
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
=======
=======
>>>>>>> [Fix] MotionBlur bug fix and doctest update (#782)
>>>>>>> [Fix] MotionBlur bug fix and doctest update (#782)
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
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
=======
<<<<<<< master
<<<<<<< refs/remotes/kornia/master
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
<<<<<<< master
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
<<<<<<< refs/remotes/kornia/master
=======
<<<<<<< master
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
=======
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
<<<<<<< refs/remotes/kornia/master
=======
=======
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
>>>>>>> [FIX] fix device issue for get_motion_kernel2d (#775)
=======
>>>>>>> Exposed rng generation device and dtype for augmentations. (#770)
    return (device, dtype)
