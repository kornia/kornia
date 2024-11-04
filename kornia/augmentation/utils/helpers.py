from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.distributions import Beta, Uniform

from kornia.core import Tensor, as_tensor
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.utils import _extract_device_dtype


def _validate_input(f: Callable[..., Any]) -> Callable[..., Any]:
    r"""Validate the 2D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

        _validate_shape(input.shape, required_shapes=("BCHW",))
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _validate_input3d(f: Callable[..., Any]) -> Callable[..., Any]:
    r"""Validate the 3D input of the wrapped function.

    Args:
        f: a function that takes the first argument as tensor.

    Returns:
        the wrapped function after input is validated.
    """

    @wraps(f)
    def wrapper(input: Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

        input_shape = len(input.shape)
        if input_shape != 5:
            raise AssertionError(f"Expect input of 5 dimensions, got {input_shape} instead")
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _infer_batch_shape(input: Union[Tensor, Tuple[Tensor, Tensor]]) -> torch.Size:
    r"""Infer input shape.

    Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input(input[0])
    else:
        tensor = _transform_input(input)
    return tensor.shape


def _infer_batch_shape3d(input: Union[Tensor, Tuple[Tensor, Tensor]]) -> torch.Size:
    r"""Infer input shape.

    Input may be either (tensor,) or (tensor, transform_matrix)
    """
    if isinstance(input, tuple):
        tensor = _transform_input3d(input[0])
    else:
        tensor = _transform_input3d(input)
    return tensor.shape


def _transform_input_by_shape(input: Tensor, reference_shape: Tensor, match_channel: bool = True) -> Tensor:
    """Reshape an input tensor to have the same dimensions as the reference_shape.

    Arguments
        input: tensor to be transformed
        reference_shape: shape used as reference
        match_channel: if True, C_{src} == C_{ref}. otherwise, no constrain. C =1 by default
    """
    B = reference_shape[-4] if len(reference_shape) >= 4 else None
    C = reference_shape[-3] if len(reference_shape) >= 3 else None

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        # If the first dim matches within the batch_size, add a `C` dim
        # Useful to handler Masks without `C` dimensions
        input = input.unsqueeze(1) if B == input.shape[-3] else input.unsqueeze(0)

    if match_channel and C:
        if not input.shape[-3] == C:
            raise ValueError("The C dimension of tensor did not match with the reference tensor.")
    elif match_channel and C is None:
        raise ValueError("The reference tensor do not have a C dimension!")

    return input


def _transform_input3d_by_shape(input: Tensor, reference_shape: Tensor, match_channel: bool = True) -> Tensor:
    """Reshape an input tensor to have the same dimensions as the reference_shape.

    Arguments
        input: tensor to be transformed
        reference_shape: shape used as reference
        match_channel: if True, C_{src} == C_{ref}. otherwise, no constrain. C =1 by default
    """
    B = reference_shape[-5] if len(reference_shape) >= 5 else None
    C = reference_shape[-4] if len(reference_shape) >= 4 else None

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) == 4 and B == input.shape[-4]:
        # If the first dim matches within the batch_size, add a `C` dim
        # Useful to handler Masks without `C` dimensions
        input = input.unsqueeze(2)

    if match_channel and C:
        if not input.shape[-4] == C:
            raise ValueError("The C dimension of tensor did not match with the reference tensor.")
    elif match_channel and C is None:
        raise ValueError("The reference tensor do not have a C dimension!")

    return input


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


def _validate_input_dtype(input: Tensor, accepted_dtypes: List[torch.dtype]) -> None:
    r"""Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        input: Tensor
        accepted_dtypes: List. e.g. [torch.float32, torch.float64]
    """
    if input.dtype not in accepted_dtypes:
        raise TypeError(f"Expected input of {accepted_dtypes}. Got {input.dtype}")


def _transform_output_shape(
    output: Tensor, shape: Tuple[int, ...], *, reference_shape: Optional[Tensor] = None
) -> Tensor:
    r"""Collapse the broadcasted batch dimensions an input tensor to be the specified shape.
    Args:
        input: Tensor
        shape: List/tuple of int

    Returns:
        Tensor
    """
    out_tensor = output.clone()

    for dim in range(len(out_tensor.shape) - len(shape)):
        idx = 0
        if reference_shape is not None and out_tensor.shape[0] == reference_shape[0] != 1 and len(shape) > 2:
            idx = 1
        if out_tensor.shape[idx] != 1:
            raise AssertionError(f"Dimension {dim} of input is expected to be 1, got {out_tensor.shape[idx]}")
        out_tensor = out_tensor.squeeze(idx)

    return out_tensor


def _validate_shape(shape: Union[Tuple[int, ...], torch.Size], required_shapes: Tuple[str, ...] = ("BCHW",)) -> None:
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
    r"""Validate if an input has the right shape.

    e.g. to check if an input is channel first.
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
    shape: Union[Tuple[int, ...], torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch: Optional[bool] = False,
) -> Tensor:
    r"""The uniform reparameterized sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.
    """
    if isinstance(shape, tuple):
        shape = torch.Size(shape)

    if same_on_batch:
        rsample_size = torch.Size((1, *shape[1:]))
        rsample = dist.rsample(rsample_size)
        return rsample.repeat(shape[0], *[1] * (len(rsample.shape) - 1))
    return dist.rsample(shape)


def _adapted_sampling(
    shape: Union[Tuple[int, ...], torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch: Optional[bool] = False,
) -> Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.
    """
    if isinstance(shape, tuple):
        shape = torch.Size(shape)

    if same_on_batch:
        return dist.sample(torch.Size((1, *shape[1:]))).repeat(shape[0], *[1] * (len(shape) - 1))
    return dist.sample(shape)


def _adapted_uniform(
    shape: Union[Tuple[int, ...], torch.Size],
    low: Union[float, Tensor],
    high: Union[float, Tensor],
    same_on_batch: bool = False,
) -> Tensor:
    r"""The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If low/high is a tensor, sampling will happen in the
    same device/dtype as low/high tensor.
    """
    device, dtype = _extract_device_dtype(
        [low if isinstance(low, Tensor) else None, high if isinstance(high, Tensor) else None]
    )
    low = as_tensor(low, device=device, dtype=dtype)
    high = as_tensor(high, device=device, dtype=dtype)
    # validate_args=False to fix pytorch 1.7.1 error:
    #     ValueError: Uniform is not defined when low>= high.
    dist = Uniform(low, high, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _adapted_beta(
    shape: Union[Tuple[int, ...], torch.Size],
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    same_on_batch: bool = False,
) -> Tensor:
    r"""The beta sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If a/b is a tensor, sampling will happen in the same
    device/dtype as a/b tensor.
    """
    device, dtype = _extract_device_dtype([a if isinstance(a, Tensor) else None, b if isinstance(b, Tensor) else None])
    a = as_tensor(a, device=device, dtype=dtype)
    b = as_tensor(b, device=device, dtype=dtype)
    dist = Beta(a, b, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _shape_validation(param: Tensor, shape: Union[Tuple[int, ...], List[int]], name: str) -> None:
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
    params: Dict[str, Any],
    params_override: Optional[Dict[str, Any]] = None,
    if_none_exist: str = "ignore",
    in_place: bool = False,
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
        elif if_none_exist == "ignore":
            pass
        elif if_none_exist == "raise":
            raise RuntimeError(f"Param `{k}` not existed in `{params_override}`.")
        else:
            raise ValueError(f"`{if_none_exist}` is not a valid option.")
    return out


def preprocess_boxes(input: Union[Tensor, Boxes], mode: str = "vertices_plus") -> Boxes:
    r"""Preprocess input boxes.

    Args:
        input: 2D boxes, shape of :math:`(N, 4, 2)`, :math:`(B, N, 4, 2)` or a list of :math:`(N, 4, 2)`.
            See below for more details.
        mode: The format in which the boxes are provided.

            * 'xyxy': boxes are assumed to be in the format ``xmin, ymin, xmax, ymax`` where ``width = xmax - xmin``
                and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
            * 'xyxy_plus': similar to 'xyxy' mode but where box width and length are defined as
                ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``.
                With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
            * 'xywh': boxes are assumed to be in the format ``xmin, ymin, width, height`` where
                ``width = xmax - xmin`` and ``height = ymax - ymin``. With shape :math:`(N, 4)`, :math:`(B, N, 4)`.
            * 'vertices': boxes are defined by their vertices points in the following ``clockwise`` order:
                *top-left, top-right, bottom-right, bottom-left*. Vertices coordinates are in (x,y) order. Finally,
                box width and height are defined as ``width = xmax - xmin`` and ``height = ymax - ymin``.
                With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.
            * 'vertices_plus': similar to 'vertices' mode but where box width and length are defined as
                ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. ymin + 1``.
                With shape :math:`(N, 4, 2)` or :math:`(B, N, 4, 2)`.

    Note:
        **2D boxes format** is defined as a floating data type tensor of shape ``Nx4x2`` or ``BxNx4x2``
        where each box is a `quadrilateral <https://en.wikipedia.org/wiki/Quadrilateral>`_ defined by it's 4 vertices
        coordinates (A, B, C, D). Coordinates must be in ``x, y`` order. The height and width of a box is defined as
        ``width = xmax - xmin + 1`` and ``height = ymax - ymin + 1``. Examples of
        `quadrilaterals <https://en.wikipedia.org/wiki/Quadrilateral>`_ are rectangles, rhombus and trapezoids.
    """
    # TODO: We may allow list here.
    # input is BxNx4x2 or Boxes.
    if isinstance(input, Tensor):
        if not (len(input.shape) == 4 and input.shape[2:] == torch.Size([4, 2])):
            raise RuntimeError(f"Only BxNx4x2 tensor is supported. Got {input.shape}.")
        input = Boxes.from_tensor(input, mode=mode)
    if not isinstance(input, Boxes):
        raise RuntimeError(f"Expect `Boxes` type. Got {type(input)}.")
    return input


def preprocess_keypoints(input: Union[Tensor, Keypoints]) -> Keypoints:
    """Preprocess input keypoints."""
    # TODO: We may allow list here.
    if isinstance(input, Tensor):
        if not (len(input.shape) == 3 and input.shape[1:] == torch.Size([2])):
            raise RuntimeError(f"Only BxNx2 tensor is supported. Got {input.shape}.")
        input = Keypoints(input, False)
    if isinstance(input, Keypoints):
        raise RuntimeError(f"Expect `Keypoints` type. Got {type(input)}.")
    return input


def preprocess_classes(input: Tensor) -> Tensor:
    """Preprocess input class tags."""
    # TODO: We may allow list here.
    return input


class MultiprocessWrapper:
    """Utility class which when used as a base class, makes the class work with the 'spawn' multiprocessing
    context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        args = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}

        super().__init__(*args, **kwargs)
