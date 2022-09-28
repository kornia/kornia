from __future__ import annotations

import torch

from kornia.augmentation.utils import _adapted_uniform, _common_param_check, _joint_range_check
from kornia.utils.helpers import _deprecated, _extract_device_dtype


@_deprecated()
def random_rotation_generator(
    batch_size: int,
    degrees: torch.Tensor,
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Get parameters for ``rotate`` for a random rotate transform.

    Args:
        batch_size (int): the tensor batch size.
        degrees (torch.Tensor): range of degrees with shape (2) to select from.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - degrees (torch.Tensor): element-wise rotation degrees with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(degrees, "degrees")

    _degrees = _adapted_uniform(
        (batch_size,),
        degrees[0].to(device=device, dtype=dtype),
        degrees[1].to(device=device, dtype=dtype),
        same_on_batch,
    )
    _degrees = _degrees.to(device=degrees.device, dtype=degrees.dtype)

    return dict(degrees=_degrees)


@_deprecated()
def random_solarize_generator(
    batch_size: int,
    thresholds: torch.Tensor = torch.tensor([0.4, 0.6]),
    additions: torch.Tensor = torch.tensor([-0.1, 0.1]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Generate random solarize parameters for a batch of images.

    For each pixel in the image less than threshold, we add 'addition' amount to it and then clip the pixel value
    to be between 0 and 1.0

    Args:
        batch_size (int): the number of images.
        thresholds (torch.Tensor): Pixels less than threshold will selected. Otherwise, subtract 1.0 from the pixel.
            Takes in a range tensor of (0, 1). Default value will be sampled from [0.4, 0.6].
        additions (torch.Tensor): The value is between -0.5 and 0.5. Default value will be sampled from [-0.1, 0.1]
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - thresholds_factor (torch.Tensor): element-wise thresholds factors with a shape of (B,).
            - additions_factor (torch.Tensor): element-wise additions factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(thresholds, 'thresholds', (0, 1))
    _joint_range_check(additions, 'additions', (-0.5, 0.5))

    _device, _dtype = _extract_device_dtype([thresholds, additions])

    thresholds_factor = _adapted_uniform(
        (batch_size,),
        thresholds[0].to(device=device, dtype=dtype),
        thresholds[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    additions_factor = _adapted_uniform(
        (batch_size,),
        additions[0].to(device=device, dtype=dtype),
        additions[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    return dict(
        thresholds_factor=thresholds_factor.to(device=_device, dtype=_dtype),
        additions_factor=additions_factor.to(device=_device, dtype=_dtype),
    )


@_deprecated()
def random_sharpness_generator(
    batch_size: int,
    sharpness: torch.Tensor = torch.tensor([0, 1.0]),
    same_on_batch: bool = False,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    r"""Generate random sharpness parameters for a batch of images.

    Args:
        batch_size (int): the number of images.
        sharpness (torch.Tensor): Must be above 0. Default value is sampled from (0, 1).
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        device (torch.device): the device on which the random numbers will be generated. Default: cpu.
        dtype (torch.dtype): the data type of the generated random numbers. Default: float32.

    Returns:
        params Dict[str, torch.Tensor]: parameters to be passed for transformation.
            - sharpness_factor (torch.Tensor): element-wise sharpness factors with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes.
    """
    _common_param_check(batch_size, same_on_batch)
    _joint_range_check(sharpness, 'sharpness', bounds=(0, float('inf')))

    sharpness_factor = _adapted_uniform(
        (batch_size,),
        sharpness[0].to(device=device, dtype=dtype),
        sharpness[1].to(device=device, dtype=dtype),
        same_on_batch,
    )

    return dict(sharpness_factor=sharpness_factor.to(device=sharpness.device, dtype=sharpness.dtype))
