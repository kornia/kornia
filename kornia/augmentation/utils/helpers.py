from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.distributions import Beta, Uniform

from kornia.utils import _extract_device_dtype


def _validate_input(f: Callable) -> Callable:
    r"""Validate the 2D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: Tensor, *args, **kwargs):
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

        _validate_shape(input.shape, required_shapes=('BCHW',))
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _validate_input3d(f: Callable) -> Callable:
    r"""Validate the 3D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: Tensor, *args, **kwargs):
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

        input_shape = len(input.shape)
        if input_shape != 5:
            raise AssertionError(f'Expect input of 5 dimensions, got {input_shape} instead')
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _infer_batch_shape(input: Union[Tensor, Tuple[Tensor, Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)"""
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


def _infer_batch_shape3d(input: Union[Tensor, Tuple[Tensor, Tensor]]) -> torch.Size:
    r"""Infer input shape. Input may be either (tensor,) or (tensor, transform_matrix)"""
    if isinstance(input, tuple):
        tensor = _transform_input3d(input[0])
    else:
        tensor = _transform_input3d(input)
    return tensor.shape


def _transform_input(input: Tensor) -> Tensor:
    r"""Reshape an input tensor to be (*, C, H, W). Accept either (H, W), (C, H, W) or (*, C, H, W).
    Args:
        input: Tensor

    Returns:
        Tensor
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if len(input.shape) not in [2, 3, 4]:
        raise ValueError(f"Input size must have a shape of either (H, W), (C, H, W) or (*, C, H, W). Got {input.shape}")

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    return input


def _transform_input3d(input: Tensor) -> Tensor:
    r"""Reshape an input tensor to be (*, C, D, H, W). Accept either (D, H, W), (C, D, H, W) or (*, C, D, H, W).
    Args:
        input: Tensor

    Returns:
        Tensor
    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if len(input.shape) not in [3, 4, 5]:
        raise ValueError(
            f"Input size must have a shape of either (D, H, W), (C, D, H, W) or (*, C, D, H, W). Got {input.shape}"
        )

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) == 4:
        input = input.unsqueeze(0)

    return input


def _validate_input_dtype(input: Tensor, accepted_dtypes: List) -> None:
    r"""Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        input: Tensor
        accepted_dtypes: List. e.g. [torch.float32, torch.float64]
    """
    if input.dtype not in accepted_dtypes:
        raise TypeError(f"Expected input of {accepted_dtypes}. Got {input.dtype}")


def _transform_output_shape(
    output: Tensor, shape: Tuple
) -> Tensor:
    r"""Collapse the broadcasted batch dimensions an input tensor to be the specified shape.
    Args:
        input: Tensor
        shape: List/tuple of int

    Returns:
        Tensor
    """
    out_tensor: Tensor
    out_tensor = cast(Tensor, output)

    for dim in range(len(out_tensor.shape) - len(shape)):
        if out_tensor.shape[0] != 1:
            raise AssertionError(f'Dimension {dim} of input is ' f'expected to be 1, got {out_tensor.shape[0]}')
        out_tensor = out_tensor.squeeze(0)

    return out_tensor  # type: ignore


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


def _validate_input_shape(input: Tensor, channel_index: int, number: int) -> bool:
    r"""Validate if an input has the right shape. e.g. to check if an input is channel first.
    If channel first, the second channel of an RGB input shall be fixed to 3. To verify using:
        _validate_input_shape(input, 1, 3)
    Args:
        input: Tensor
        channel_index: int
        number: int
    Returns:
        bool
    """
    return input.shape[channel_index] == number


def _adapted_rsampling(
    shape: Union[Tuple, torch.Size], dist: torch.distributions.Distribution, same_on_batch=False
) -> Tensor:
    r"""The uniform reparameterized sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.rsample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    return dist.rsample(shape)


def _adapted_sampling(
    shape: Union[Tuple, torch.Size], dist: torch.distributions.Distribution, same_on_batch=False
) -> Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(shape[0], *[1] * (len(shape) - 1))
    return dist.sample(shape)


def _adapted_uniform(
    shape: Union[Tuple, torch.Size],
    low: Union[float, int, Tensor],
    high: Union[float, int, Tensor],
    same_on_batch: bool = False,
) -> Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If low/high is a tensor, sampling will happen
    in the same device/dtype as low/high tensor.
    """
    device, dtype = _extract_device_dtype(
        [low if isinstance(low, Tensor) else None, high if isinstance(high, Tensor) else None]
    )
    low = torch.as_tensor(low, device=device, dtype=dtype)
    high = torch.as_tensor(high, device=device, dtype=dtype)
    # validate_args=False to fix pytorch 1.7.1 error:
    #     ValueError: Uniform is not defined when low>= high.
    dist = Uniform(low, high, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _adapted_beta(
    shape: Union[Tuple, torch.Size],
    a: Union[float, int, Tensor],
    b: Union[float, int, Tensor],
    same_on_batch: bool = False,
) -> Tensor:
    r"""The beta sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If a/b is a tensor, sampling will happen
    in the same device/dtype as a/b tensor.
    """
    device, dtype = _extract_device_dtype(
        [a if isinstance(a, Tensor) else None, b if isinstance(b, Tensor) else None]
    )
    a = torch.as_tensor(a, device=device, dtype=dtype)
    b = torch.as_tensor(b, device=device, dtype=dtype)
    dist = Beta(a, b, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _shape_validation(param: Tensor, shape: Union[tuple, list], name: str) -> None:
    if param.shape != torch.Size(shape):
        raise AssertionError(f"Invalid shape for {name}. Expected {shape}. Got {param.shape}")


def deepcopy_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform deep copy on any dict.

    Support tensor copying here.
    """
    out = {}
    for k, v in params.items():
        # NOTE: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol
        if isinstance(v, Tensor):
            out.update({k: v.clone()})
        else:
            out.update({k: v})
    return out


def override_parameters(
    params: Dict[str, Any], params_override: Optional[Dict[str, Any]] = None,
    if_none_exist: str = 'ignore', in_place: bool = False
) -> Dict[str, Any]:
    """Override params dict w.r.t params_override.

    Args:
        params: source parameters.
        params_override: key-values to override the source parameters.
        if_none_exist: behaviour if the key in `params_override` does not exist in `params`.
            'raise' | 'ignore'.
        in_place: if to override in-place or not.
    """

    if params_override is None:
        return params
    out = params if in_place else deepcopy_dict(params)
    for k, v in params_override.items():
        if k in params_override:
            out[k] = v
        else:
            if if_none_exist == 'ignore':
                pass
            elif if_none_exist == 'raise':
                raise RuntimeError(f"Param `{k}` not existed in `{params_override}`.")
            else:
                raise ValueError(f"`{if_none_exist}` is not a valid option.")
    return out
