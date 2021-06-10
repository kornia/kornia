from typing import cast, List, Optional, Tuple, Union

import torch


def _common_param_check(batch_size: int, same_on_batch: Optional[bool] = None):
    """Valid batch_size and same_on_batch params."""
    assert type(batch_size) is int and batch_size >= 0, f"`batch_size` shall be a positive integer. Got {batch_size}."
    if same_on_batch is not None:
        assert type(same_on_batch) is bool, f"`same_on_batch` shall be boolean. Got {same_on_batch}."


def _range_bound(
    factor: Union[torch.Tensor, float, Tuple[float, float], List[float]],
    name: str,
    center: float = 0.0,
    bounds: Tuple[float, float] = (0, float('inf')),
    check: Optional[str] = 'joint',
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    r"""Check inputs and compute the corresponding factor bounds"""
    if not isinstance(factor, (torch.Tensor)):
        factor = torch.tensor(factor, device=device, dtype=dtype)
    factor_bound: torch.Tensor

    if factor.dim() == 0:
        if factor < 0:
            raise ValueError(f"If {name} is a single number number, it must be non negative. Got {factor}")
        # Should be something other than clamp
        # Currently, single value factor will not out of scope as long as the user provided it.
        # Note: I personally think throw an error will be better than a coarse clamp.
        factor_bound = factor.repeat(2) * torch.tensor([-1.0, 1.0], device=factor.device, dtype=factor.dtype) + center
        factor_bound = factor_bound.clamp(bounds[0], bounds[1])
    else:
        factor_bound = torch.as_tensor(factor, device=device, dtype=dtype)

    if check is not None:
        if check == 'joint':
            _joint_range_check(factor_bound, name, bounds)
        elif check == 'singular':
            _singular_range_check(factor_bound, name, bounds)
        else:
            raise NotImplementedError(f"methods '{check}' not implemented.")

    return factor_bound


def _joint_range_check(ranged_factor: torch.Tensor, name: str, bounds: Optional[Tuple[float, float]] = None) -> None:
    """check if bounds[0] <= ranged_factor[0] <= ranged_factor[1] <= bounds[1]"""
    if bounds is None:
        bounds = (float('-inf'), float('inf'))
    if ranged_factor.dim() == 1 and len(ranged_factor) == 2:
        if not bounds[0] <= ranged_factor[0] or not bounds[1] >= ranged_factor[1]:
            raise ValueError(f"{name} out of bounds. Expected inside {bounds}, got {ranged_factor}.")

        if not bounds[0] <= ranged_factor[0] <= ranged_factor[1] <= bounds[1]:
            raise ValueError(f"{name}[0] should be smaller than {name}[1] got {ranged_factor}")
    else:
        raise TypeError(
            f"{name} should be a tensor with length 2 whose values between {bounds}. " f"Got {ranged_factor}."
        )


def _singular_range_check(
    ranged_factor: torch.Tensor,
    name: str,
    bounds: Optional[Tuple[float, float]] = None,
    skip_none: bool = False,
    mode: str = '2d',
) -> None:
    """check if bounds[0] <= ranged_factor[0] <= bounds[1] and bounds[0] <= ranged_factor[1] <= bounds[1]"""
    if mode == '2d':
        dim_size = 2
    elif mode == '3d':
        dim_size = 3
    else:
        raise ValueError(f"'mode' shall be either 2d or 3d. Got {mode}")

    if skip_none and ranged_factor is None:
        return
    if bounds is None:
        bounds = (float('-inf'), float('inf'))
    if ranged_factor.dim() == 1 and len(ranged_factor) == dim_size:
        for f in ranged_factor:
            if not bounds[0] <= f <= bounds[1]:
                raise ValueError(f"{name} out of bounds. Expected inside {bounds}, got {ranged_factor}.")
    else:
        raise TypeError(
            f"{name} should be a float number or a tuple with length {dim_size} whose values between {bounds}."
            f"Got {ranged_factor}"
        )


def _tuple_range_reader(
    input_range: Union[torch.Tensor, float, tuple],
    target_size: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Given target_size, it will generate the corresponding (target_size, 2) range tensor for element-wise params.

    Example:
    >>> degree = torch.tensor([0.2, 0.3])
    >>> _tuple_range_reader(degree, 3)  # read degree for yaw, pitch and roll.
    tensor([[0.2000, 0.3000],
            [0.2000, 0.3000],
            [0.2000, 0.3000]])
    """
    target_shape = torch.Size([target_size, 2])
    if not torch.is_tensor(input_range):
        if isinstance(input_range, (float, int)):
            if input_range < 0:
                raise ValueError(f"If input_range is only one number it must be a positive number. Got{input_range}")
            input_range_tmp = torch.tensor([-input_range, input_range], device=device, dtype=dtype).repeat(
                target_shape[0], 1
            )

        elif (
            isinstance(input_range, (tuple, list))
            and len(input_range) == 2
            and isinstance(input_range[0], (float, int))
            and isinstance(input_range[1], (float, int))
        ):
            input_range_tmp = torch.tensor(input_range, device=device, dtype=dtype).repeat(target_shape[0], 1)

        elif (
            isinstance(input_range, (tuple, list))
            and len(input_range) == target_shape[0]
            and all(isinstance(x, (float, int)) for x in input_range)
        ):
            input_range_tmp = torch.tensor([(-s, s) for s in input_range], device=device, dtype=dtype)

        elif (
            isinstance(input_range, (tuple, list))
            and len(input_range) == target_shape[0]
            and all(isinstance(x, (tuple, list)) for x in input_range)
        ):
            input_range_tmp = torch.tensor(input_range, device=device, dtype=dtype)

        else:
            raise TypeError(
                "If not pass a tensor, it must be float, (float, float) for isotropic operation or a tuple of "
                f"{target_size} floats or {target_size} (float, float) for independent operation. Got {input_range}."
            )

    else:
        # https://mypy.readthedocs.io/en/latest/casts.html cast to please mypy gods
        input_range = cast(torch.Tensor, input_range)
        if (len(input_range.shape) == 0) or (len(input_range.shape) == 1 and len(input_range) == 1):
            if input_range < 0:
                raise ValueError(f"If input_range is only one number it must be a positive number. Got{input_range}")
            input_range_tmp = input_range.repeat(2) * torch.tensor(
                [-1.0, 1.0], device=input_range.device, dtype=input_range.dtype
            )
            input_range_tmp = input_range_tmp.repeat(target_shape[0], 1)

        elif len(input_range.shape) == 1 and len(input_range) == 2:
            input_range_tmp = input_range.repeat(target_shape[0], 1)

        elif len(input_range.shape) == 1 and len(input_range) == target_shape[0]:
            input_range_tmp = input_range.unsqueeze(1).repeat(1, 2) * torch.tensor(
                [-1, 1], device=input_range.device, dtype=input_range.dtype
            )

        elif input_range.shape == target_shape:
            input_range_tmp = input_range

        else:
            raise ValueError(
                f"Degrees must be a {list(target_shape)} tensor for the degree range for independent operation."
                f"Got {input_range}"
            )
            input_range_tmp = input_range

    return input_range_tmp
