from functools import wraps
from typing import Callable, cast, List, Optional, Tuple, Union

import torch
from torch.distributions import Beta, Uniform

from kornia.utils import _extract_device_dtype


def _validate_input(f: Callable) -> Callable:
    r"""Validates the 2D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: torch.Tensor, *args, **kwargs):
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        _validate_shape(input.shape, required_shapes=('BCHW',))
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _validate_input3d(f: Callable) -> Callable:
    r"""Validates the 3D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: torch.Tensor, *args, **kwargs):
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        input_shape = len(input.shape)
        assert input_shape == 5, f'Expect input of 5 dimensions, got {input_shape} instead'
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _infer_batch_shape(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)"""
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


def _infer_batch_shape3d(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)"""
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
        raise ValueError(f"Input size must have a shape of either (H, W), (C, H, W) or (*, C, H, W). Got {input.shape}")

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
            f"Input size must have a shape of either (D, H, W), (C, D, H, W) or (*, C, D, H, W). Got {input.shape}"
        )

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


def _transform_output_shape(
    output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], shape: Tuple
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Collapse the broadcasted batch dimensions an input tensor to be the specified shape.
    Args:
        input: torch.Tensor
        shape: List/tuple of int

    Returns:
        torch.Tensor
    """
    is_tuple = isinstance(output, tuple)
    out_tensor: torch.Tensor
    trans_matrix: Optional[torch.Tensor]
    if is_tuple:
        out_tensor, trans_matrix = cast(Tuple[torch.Tensor, torch.Tensor], output)
    else:
        out_tensor = cast(torch.Tensor, output)
        trans_matrix = None

    if trans_matrix is not None:
        if len(out_tensor.shape) > len(shape):  # if output is broadcasted
            assert trans_matrix.shape[0] == 1, (
                f'Dimension 0 of transformation matrix is ' f'expected to be 1, got {trans_matrix.shape[0]}'
            )
        trans_matrix = trans_matrix.squeeze(0)

    for dim in range(len(out_tensor.shape) - len(shape)):
        assert out_tensor.shape[0] == 1, f'Dimension {dim} of input is ' f'expected to be 1, got {out_tensor.shape[0]}'
        out_tensor = out_tensor.squeeze(0)

    return (out_tensor, trans_matrix) if is_tuple else out_tensor  # type: ignore


def _validate_shape(shape: Union[Tuple, torch.Size], required_shapes: Tuple[str, ...] = ("BCHW",)) -> None:
    r"""Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        shape: tensor shape
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


def _adapted_rsampling(
    shape: Union[Tuple, torch.Size], dist: torch.distributions.Distribution, same_on_batch=False
) -> torch.Tensor:
    r"""The uniform reparameterized sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.rsample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    return dist.rsample(shape)


def _adapted_sampling(
    shape: Union[Tuple, torch.Size], dist: torch.distributions.Distribution, same_on_batch=False
) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    return dist.sample(shape)


def _adapted_uniform(
    shape: Union[Tuple, torch.Size],
    low: Union[float, int, torch.Tensor],
    high: Union[float, int, torch.Tensor],
    same_on_batch: bool = False,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If low/high is a tensor, sampling will happen
    in the same device/dtype as low/high tensor.
    """
    device, dtype = _extract_device_dtype(
        [low if isinstance(low, torch.Tensor) else None, high if isinstance(high, torch.Tensor) else None]
    )
    low = torch.as_tensor(low, device=device, dtype=dtype)
    high = torch.as_tensor(high, device=device, dtype=dtype)
    # validate_args=False to fix pytorch 1.7.1 error:
    #     ValueError: Uniform is not defined when low>= high.
    dist = Uniform(low, high, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _adapted_beta(
    shape: Union[Tuple, torch.Size],
    a: Union[float, int, torch.Tensor],
    b: Union[float, int, torch.Tensor],
    same_on_batch: bool = False,
) -> torch.Tensor:
    r"""The beta sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If a/b is a tensor, sampling will happen
    in the same device/dtype as a/b tensor.
    """
    device, dtype = _extract_device_dtype(
        [a if isinstance(a, torch.Tensor) else None, b if isinstance(b, torch.Tensor) else None]
    )
    a = torch.as_tensor(a, device=device, dtype=dtype)
    b = torch.as_tensor(b, device=device, dtype=dtype)
    dist = Beta(a, b, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _shape_validation(param: torch.Tensor, shape: Union[tuple, list], name: str) -> None:
    assert param.shape == torch.Size(shape), f"Invalid shape for {name}. Expected {shape}. Got {param.shape}"
