from typing import Tuple, Union, List, cast

import torch
from torch.distributions import Uniform


def _infer_batch_shape(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


def _infer_batch_shape3d(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input3d(input[0])
    else:
        tensor = _transform_input3d(input)
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


def _transform_input3d(input: torch.Tensor) -> torch.Tensor:
    r"""Reshape an input tensor to be (*, C, D, H, W). Accept either (D, H, W), (C, D, H, W) or (*, C, D, H, W).
    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) not in [3, 4, 5]:
        raise ValueError(
            f"Input size must have a shape of either (D, H, W), (C, D, H, W) or (*, C, D, H, W). Got {input.shape}")

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) == 4:
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


def _validate_shape(shape: Union[Tuple, torch.Size], required_shapes: List[str] = ["BCHW"]) -> None:
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
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low).float()
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high).float()
    dist = Uniform(low, high)
    if same_on_batch:
        return dist.rsample((1, *shape[1:])).repeat(shape[0])
    else:
        return dist.rsample(shape)


def _check_and_bound(factor: Union[float, Tuple[float, float], List[float]], name: str,
                     center: float = 0., bounds: Tuple[float, float] = (0, float('inf'))) -> Tuple[float, float]:
    r"""Check inputs and compute the corresponding factor bounds
    """
    factor_bound: Tuple[float, float]

    if isinstance(factor, (int, float)):

        if factor < 0:
            raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")

        # Should be something other than clamp
        # Currently, single value factor will not out of scope as long as the user provided it.
        factor_bound = (max(bounds[0], center - factor), min(bounds[1], center + factor))

    elif isinstance(factor, (tuple, list)) and len(factor) == 2 and \
            isinstance(factor[0], (int, float)) and isinstance(factor[1], (int, float)):

        if not bounds[0] <= factor[0] or not bounds[1] >= factor[1]:
            raise ValueError(f"{name} out of bounds. Expected inside {bounds}, got {factor}.")

        if not bounds[0] <= factor[0] <= factor[1] <= bounds[1]:
            raise ValueError(f"{name}[0] should be smaller than {name}[1] got {factor}")

        factor_bound = (factor[0], factor[1])

    else:

        raise TypeError(
            f"{name} should be a float number or a tuple with length 2 whose values between {bounds}. Got {factor}")

    return factor_bound


def _tuple_range_reader(
    input_range: Union[torch.Tensor, float, tuple],
    target_size: int
) -> torch.Tensor:
    """
    Given target_size, it will generate the correponding (target_size, 2) range tensor for tasks like
    affine transformation.
    """
    target_shape = torch.Size([target_size, 2])
    if not torch.is_tensor(input_range):
        if isinstance(input_range, (float, int)):
            if input_range < 0:
                raise ValueError(f"If input_range is only one number it must be a positive number. Got{input_range}")
            input_range_tmp = torch.tensor([-input_range, input_range]).repeat(target_shape[0], 1).to(torch.float32)

        elif isinstance(input_range, (tuple)) and len(input_range) == 2 \
                and isinstance(input_range[0], (float, int)) and isinstance(input_range[1], (float, int)):
            input_range_tmp = torch.tensor(input_range).repeat(target_shape[0], 1).to(torch.float32)

        elif isinstance(input_range, (tuple)) and len(input_range) == target_shape[0] \
                and all([isinstance(x, (float, int)) for x in input_range]):
            input_range_tmp = torch.tensor([(-s, s) for s in input_range]).to(torch.float32)

        elif isinstance(input_range, (tuple)) and len(input_range) == target_shape[0] \
                and all([isinstance(x, (tuple)) for x in input_range]):
            input_range_tmp = torch.tensor(input_range).to(torch.float32)

        else:
            raise TypeError(
                "If not pass a tensor, it must be float, (float, float) for isotropic operation or a tuple of"
                f"{target_size} floats or {target_size} (float, float) for independent operation. Got {input_range}.")

    else:
        # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
        input_range = cast(torch.Tensor, input_range)
        if len(input_range.shape) == 0:
            input_range_tmp = torch.tensor([-input_range, input_range]).repeat(target_shape[0], 1).to(torch.float32)
        elif len(input_range.shape) == 1 and len(input_range) == 1:
            input_range_tmp = torch.tensor([-input_range[0], input_range[0]]).repeat(
                target_shape[0], 1).to(torch.float32)
        elif len(input_range.shape) == 1 and len(input_range) == 2:
            input_range_tmp = input_range.repeat(target_shape[0], 1).to(torch.float32)
        elif len(input_range.shape) == 1 and len(input_range) == target_shape[0]:
            input_range_tmp = torch.tensor([(-s, s) for s in input_range]).to(torch.float32)
        elif input_range.shape == target_shape:
            input_range_tmp = input_range
        else:
            raise ValueError(
                f"Degrees must be a {list(target_shape)} tensor for the degree range for independent operation."
                f"Got {input_range}")
            input_range_tmp = input_range

    return input_range_tmp
