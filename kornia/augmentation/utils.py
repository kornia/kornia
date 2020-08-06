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


def _check_and_bound(factor: Union[torch.Tensor, float, Tuple[float, float], List[float]], name: str,
                     center: float = 0., bounds: Tuple[float, float] = (0, float('inf'))) -> torch.Tensor:
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


def _tuple_range_reader(
    shears: Union[torch.Tensor, float, tuple],
    target_size: int
) -> torch.Tensor:
    """
    Given target_size, it will generate the correponding (target_size, 2) range tensor for tasks like
    affine transformation.
    """
    target_shape = torch.Size([target_size, 2])
    if not torch.is_tensor(shears):
        if isinstance(shears, (float, int)):
            if shears < 0:
                raise ValueError(f"If shears is only one number it must be a positive number. Got{shears}")
            shears_tmp = torch.tensor([-shears, shears]).repeat(target_shape[0], 1).to(torch.float32)

        elif isinstance(shears, (tuple)) and len(shears) == 2 \
                and isinstance(shears[0], (float, int)) and isinstance(shears[1], (float, int)):
            shears_tmp = torch.tensor(shears).repeat(target_shape[0], 1).to(torch.float32)

        elif isinstance(shears, (tuple)) and len(shears) == target_shape[0] \
                and all([isinstance(x, (float, int)) for x in shears]):
            shears_tmp = torch.tensor([(-s, s) for s in shears]).to(torch.float32)

        elif isinstance(shears, (tuple)) and len(shears) == target_shape[0] \
                and all([isinstance(x, (tuple)) for x in shears]):
            shears_tmp = torch.tensor(shears).to(torch.float32)

        else:
            raise TypeError(
                "If not pass a tensor, it must be float, (float, float) for isotropic shearing or a tuple of"
                f"{target_size} floats or {target_size} (float, float) for independent shearing. Got {shears}.")

    else:
        # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
        shears = cast(torch.Tensor, shears)
        if shears.shape != target_shape:
            raise ValueError(
                f"Degrees must be a {list(target_shape)} tensor for the degree range for independent shearing."
                f"Got {shears}")
        shears_tmp = shears

    return shears_tmp
