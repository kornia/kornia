from typing import Tuple, Union, List

import torch
from torch.distributions import Uniform
from .types import (
    FloatUnionType,
    UnionType,
    UnionShape
)


def _infer_batch_shape(input: UnionType) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


def _transform_input(input: torch.Tensor) -> torch.Tensor:
    r"""Reshape an input tensor to be (*, C, H, W). Accept either (H, W), (C, H, W) or (*, C, H, W).
    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) not in [2, 3, 4]:
        raise ValueError(
            f"Input size must have a shape of either (H, W), (C, H, W) or (*, C, H, W). Got {input.shape}")

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    return input


def _validate_input_dtype(input: torch.Tensor, accepted_dtypes: List) -> None:
    r"""Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        input: torch.Tensor
        accepted_dtypes: List. e.g. [torch.float32, torch.float64]
    """
    if input.dtype not in accepted_dtypes:
        raise TypeError(f"Expected input of {accepted_dtypes}. Got {input.dtype}")


def _validate_shape(shape: UnionShape, required_shapes: List[str] = ["BCHW"]) -> None:
    r"""Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        input: torch.Tensor
        required_shapes: List. e.g. ["BCHW", "BCDHW"]
    """
    passed = False
    for required_shape in required_shapes:
        if len(shape) == len(required_shape):
            passed = True
            break
    if not passed:
        raise TypeError(f"Expected input shape in {required_shape}. Got {shape}.")


def _validate_input_shape(input: torch.Tensor, channel_index: int, number: int) -> bool:
    r"""Validate if an input has the right shape. e.g. to check if an input is channel first.
    If channel first, the second channel of an RGB input shall be fixed to 3. To verify using:
        _validate_input_shape(input, 1, 3)
    Args:
        input: torch.Tensor
        channel_index: int
        number: int
    Returns:
        bool
    """
    return input.shape[channel_index] == number


def _adapted_uniform(shape: Union[Tuple, torch.Size], low, high, same_on_batch=False) -> torch.Tensor:
    r""" The uniform function that accepts 'same_on_batch'.
    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    dist = Uniform(low, high)
    if same_on_batch:
        return dist.rsample((1, *shape[1:])).repeat(shape[0])
    else:
        return dist.rsample(shape)


def _check_and_bound(factor: FloatUnionType, name: str, center: float = 0.,
                     bounds: Tuple[float, float] = (0, float('inf'))) -> torch.Tensor:
    r"""Check inputs and compute the corresponding factor bounds
    """
    factor_bound: torch.Tensor
    if not isinstance(factor, torch.Tensor):
        factor = torch.tensor(factor, dtype=torch.float32)

    if factor.dim() == 0:
        _center = torch.tensor(center, dtype=torch.float32)

        if factor < 0:
            raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor.item()}")

        factor_bound = torch.tensor([_center - factor, _center + factor], dtype=torch.float32)
        # Should be something other than clamp
        # Currently, single value factor will not out of scope as long as the user provided it.
        factor_bound = torch.clamp(factor_bound, bounds[0], bounds[1])

    elif factor.shape[0] == 2 and factor.dim() == 1:

        if not bounds[0] <= factor[0] or not bounds[1] >= factor[1]:
            raise ValueError(f"{name} out of bounds. Expected inside {bounds}, got {factor}.")

        if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
            raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

        factor_bound = factor

    else:

        raise TypeError(
            f"The {name} should be a float number or a tuple with length 2 whose values move between {bounds}.")

    return factor_bound
