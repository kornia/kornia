# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import wraps
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Uniform

# Both moved to kornia.core.ops; re-exported so that existing imports from this module keep working.
from kornia.core.exceptions import InputRankError
from kornia.core.ops import _constant_tensor, _flatten_constant  # noqa: F401
from kornia.core.utils import _extract_device_dtype
from kornia.geometry.boxes import Boxes, VideoBoxes
from kornia.geometry.keypoints import Keypoints, VideoKeypoints


def _pad_with_fill(
    input: torch.Tensor, padding: List[int], fill: Union[float, Sequence[float]], mode: str
) -> torch.Tensor:
    """Pad ``input`` with a scalar or with one constant per channel.

    ``torch.nn.functional.pad`` only accepts a scalar ``value``, so a sequence ``fill`` is applied by padding
    with zeros and writing the per-channel constants into the region the padding added. A sequence is only
    meaningful for ``mode="constant"``.
    """
    if isinstance(fill, (int, float)):
        return F.pad(input, padding, value=float(fill), mode=mode)

    if mode != "constant":
        raise ValueError(f"A sequence `fill` needs `padding_mode='constant'`, got '{mode}'.")

    channels = input.shape[1]
    # Built in-graph rather than lifted from a host list, so a compiled CUDA call issues no host-device copy.
    if isinstance(fill, (list, tuple)):
        values = _constant_tensor(fill, device=input.device, dtype=input.dtype).flatten()
    else:
        values = torch.as_tensor(fill, device=input.device, dtype=input.dtype).flatten()
    if values.numel() != channels:
        raise ValueError(f"`fill` must hold one value per channel: got {values.numel()} for {channels} channels.")

    padded = F.pad(input, padding, value=0.0, mode=mode)
    # 1 where a voxel came from the input, 0 where the padding added one. Negative padding crops both alike.
    inside = F.pad(torch.ones_like(input[:, :1]), padding, value=0.0, mode=mode)
    return torch.where(inside.bool(), padded, values.view(1, channels, *([1] * (padded.ndim - 2))))


def _boxes_to_padded_tensor(boxes: Boxes, mode: str) -> torch.Tensor:
    """Export ``boxes`` as one dense tensor whose trailing padding rows are exactly zero.

    :class:`~kornia.geometry.boxes.Boxes` records per-sample trailing padding in ``_N`` and its default export
    returns a ragged list for such an object. The padding rows are zeroed after the export rather than stored as
    zeros, because a zero stored box exports as ``[0, 0, 1, 1]`` in the exclusive ``'xyxy'``, ``'xywh'`` and
    ``'vertices'`` modes, which reads as a real one-pixel box.
    """
    out = boxes.to_tensor(mode, as_padded_sequence=True)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Expected a padded tensor export. Got {type(out)}.")
    if boxes._N is None:
        return out
    num_boxes = out.shape[1]
    real = torch.tensor([num_boxes - n for n in boxes._N], device=out.device)
    valid = torch.arange(num_boxes, device=out.device)[None] < real[:, None]
    valid = valid.reshape(valid.shape + (1,) * (out.dim() - 2))
    return torch.where(valid, out, torch.zeros((), device=out.device, dtype=out.dtype))


def _frame_boxes(boxes: VideoBoxes) -> Boxes:
    r"""View :class:`VideoBoxes` as the frame-level :class:`Boxes` of its :math:`(B \cdot T, N, 4, 2)` storage.

    The children of :class:`VideoSequential` run on the flattened frames, and their parameters (a mix's
    ``permutation``, a gate) index frames. ``VideoBoxes`` indexes whole videos, so a child given one would select
    videos with frame indices (#6173). The view shares the storage and the validity mask.
    """
    frames = Boxes(boxes._data, False, mode=boxes._mode)
    frames._N = boxes._N
    frames._valid = boxes._valid
    frames._is_batched = boxes._is_batched
    return frames


def _video_boxes(frames: Boxes, temporal_channel_size: int) -> VideoBoxes:
    """Wrap frame-level :class:`Boxes` back into :class:`VideoBoxes` with ``temporal_channel_size`` frames.

    A ``VideoBoxes`` is a dense container, so the per-frame padding counts ``_N`` are not carried over. Padding a
    child adds, such as the rows a partial ``RandomMosaic`` gate pads a skipped frame with, is marked invalid
    instead; every export zeroes invalid rows and every indexed read or write carries them.
    """
    boxes = VideoBoxes(frames._data, False, mode=frames._mode)
    boxes._valid = frames._valid
    boxes._is_batched = frames._is_batched
    boxes.temporal_channel_size = temporal_channel_size
    return boxes


def _frame_keypoints(keypoints: VideoKeypoints) -> Keypoints:
    r"""View :class:`VideoKeypoints` as the frame-level :class:`Keypoints` of its :math:`(B \cdot T, N, 2)` storage.

    As :func:`_frame_boxes` does for boxes: the children of :class:`VideoSequential` index frames, and
    ``VideoKeypoints`` indexes whole videos. The view shares the storage.
    """
    frames = Keypoints(keypoints._data, False)
    frames._is_batched = keypoints._is_batched
    return frames


def _video_keypoints(frames: Keypoints, temporal_channel_size: int) -> VideoKeypoints:
    """Wrap frame-level :class:`Keypoints` back into :class:`VideoKeypoints` with ``temporal_channel_size`` frames."""
    keypoints = VideoKeypoints(frames._data, False)
    keypoints._is_batched = frames._is_batched
    keypoints.temporal_channel_size = temporal_channel_size
    return keypoints


def _validate_input(f: Callable[..., Any]) -> Callable[..., Any]:
    r"""Validate the 2D input of the wrapped function.

    Args:
        f: a function that takes the first argument as torch.Tensor.

    Returns:
        the wrapped function after input is validated.

    """

    @wraps(f)
    def wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        _validate_shape(input.shape, required_shapes=("BCHW",))
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _validate_input3d(f: Callable[..., Any]) -> Callable[..., Any]:
    r"""Validate the 3D input of the wrapped function.

    Args:
        f: a function that takes the first argument as torch.Tensor.

    Returns:
        the wrapped function after input is validated.

    """

    @wraps(f)
    def wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        input_shape = len(input.shape)
        if input_shape != 5:
            raise AssertionError(f"Expect input of 5 dimensions, got {input_shape} instead")
        _validate_input_dtype(input, accepted_dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64])

        return f(input, *args, **kwargs)

    return wrapper


def _infer_batch_shape(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Return the (B, C, H, W) shape of a 2D augmentation input, or of the image of an (input, transform) tuple.

    Input may be either (torch.Tensor,) or (torch.Tensor, transform_matrix)
    """
    tensor_var = _transform_input(input[0]) if isinstance(input, tuple) else _transform_input(input)
    return tensor_var.shape


def _infer_batch_shape3d(input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
    r"""Return the (B, C, D, H, W) shape of a 3D augmentation input, or of the volume of an (input, transform) tuple.

    Input may be either (torch.Tensor,) or (torch.Tensor, transform_matrix)
    """
    tensor_var = _transform_input3d(input[0]) if isinstance(input, tuple) else _transform_input3d(input)
    return tensor_var.shape


def _transform_input_by_shape(
    input: torch.Tensor, reference_shape: torch.Tensor, match_channel: bool = True
) -> torch.Tensor:
    """Add the missing batch or channel dimension of an image tensor, e.g. a mask, to match a (B, C, H, W) reference.

    Arguments:
        input: torch.Tensor to be transformed
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
            raise ValueError("The C dimension of torch.Tensor did not match with the reference torch.Tensor.")
    elif match_channel and C is None:
        raise ValueError("The reference torch.Tensor do not have a C dimension!")

    return input


def _transform_input3d_by_shape(
    input: torch.Tensor, reference_shape: torch.Tensor, match_channel: bool = True
) -> torch.Tensor:
    """Add the missing batch or channel dimension of a volume tensor, e.g. a mask, to match a (B, C, D, H, W) reference.

    A ``(D, H, W)`` input becomes ``(1, 1, D, H, W)``. A 4-dim input whose first size equals the reference batch size
    is read as ``(B, D, H, W)`` and becomes ``(B, 1, D, H, W)``; any other 4-dim input is read as ``(C, D, H, W)`` and
    becomes ``(1, C, D, H, W)``.

    Arguments:
        input: torch.Tensor to be transformed
        reference_shape: shape used as reference
        match_channel: if True, C_{src} == C_{ref}. otherwise, no constrain. C =1 by default

    """
    B = reference_shape[-5] if len(reference_shape) >= 5 else None
    C = reference_shape[-4] if len(reference_shape) >= 4 else None

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) == 4:
        # If the first dim matches within the batch_size, add a `C` dim
        # Useful to handler Masks without `C` dimensions
        input = input.unsqueeze(1) if B == input.shape[-4] else input.unsqueeze(0)

    if match_channel and C:
        if not input.shape[-4] == C:
            raise ValueError("The C dimension of torch.Tensor did not match with the reference torch.Tensor.")
    elif match_channel and C is None:
        raise ValueError("The reference torch.Tensor do not have a C dimension!")

    return input


# The input shapes an augmentation accepts, as listed in the error for any other rank (#4424).
_IMAGE_SHAPES = "either (H, W), (C, H, W) or (*, C, H, W)"
_VOLUME_SHAPES = "either (D, H, W), (C, D, H, W) or (*, C, D, H, W)"


def _raise_input_rank_error(shape: torch.Size, accepted: str) -> NoReturn:
    """Raise the :class:`~kornia.core.exceptions.InputRankError` every augmentation entry point uses for a bad rank.

    Args:
        shape: the shape of the rejected input.
        accepted: the accepted shapes, such as ``_IMAGE_SHAPES``, completing "must have a shape of ...".

    """
    raise InputRankError(f"Input size must have a shape of {accepted}. Got {shape}", actual_shape=tuple(shape))


def _transform_input(input: torch.Tensor) -> torch.Tensor:
    r"""Unsqueeze an ``(H, W)`` or ``(C, H, W)`` image tensor to 4 dims, ``(B, C, H, W)``; a 4-dim input is kept.

    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) not in [2, 3, 4]:
        _raise_input_rank_error(input.shape, _IMAGE_SHAPES)

    if len(input.shape) == 2:
        input = input.unsqueeze(0)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    return input


def _transform_input3d(input: torch.Tensor) -> torch.Tensor:
    r"""Unsqueeze a ``(D, H, W)`` or ``(C, D, H, W)`` volume to 5 dims, ``(B, C, D, H, W)``; a 5-dim input is kept.

    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor

    """
    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) not in [3, 4, 5]:
        _raise_input_rank_error(input.shape, _VOLUME_SHAPES)

    if len(input.shape) == 3:
        input = input.unsqueeze(0)

    if len(input.shape) == 4:
        input = input.unsqueeze(0)

    return input


def _validate_input_dtype(input: torch.Tensor, accepted_dtypes: List[torch.dtype]) -> None:
    r"""Raise TypeError unless the dtype of ``input`` is one of ``accepted_dtypes``.

    Args:
        input: torch.Tensor
        accepted_dtypes: List. e.g. [torch.float32, torch.float64]
    """
    if input.dtype not in accepted_dtypes:
        raise TypeError(f"Expected input of {accepted_dtypes}. Got {input.dtype}")


def _transform_output_shape(
    output: torch.Tensor, shape: Tuple[int, ...], *, reference_shape: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Collapse the broadcasted batch dimensions an input torch.Tensor to be the specified shape.

    Args:
        output: torch.Tensor
        shape: List/tuple of int
        reference_shape: torch.Tensor representation of shape to control which dimensions are collapsed.

    Returns:
        torch.Tensor

    """
    out_tensor = output

    for dim in range(len(out_tensor.shape) - len(shape)):
        idx = 0
        if reference_shape is not None and out_tensor.shape[0] == reference_shape[0] != 1 and len(shape) > 2:
            idx = 1
        if out_tensor.shape[idx] != 1:
            raise AssertionError(f"Dimension {dim} of input is expected to be 1, got {out_tensor.shape[idx]}")
        out_tensor = out_tensor.squeeze(idx)

    return out_tensor


def _validate_shape(shape: Union[Tuple[int, ...], torch.Size], required_shapes: Tuple[str, ...] = ("BCHW",)) -> None:
    r"""Raise TypeError unless ``shape`` has as many dimensions as one of ``required_shapes``, e.g. ``"BCHW"``.

    Args:
        shape: torch.Tensor shape
        required_shapes: List. e.g. ["BCHW", "BCDHW"]
    """
    passed = False
    for required_shape in required_shapes:
        if len(shape) == len(required_shape):
            passed = True
            break
    if not passed:
        raise TypeError(f"Expected input shape in {required_shapes}. Got {shape}.")


def _validate_input_shape(input: torch.Tensor, channel_index: int, number: int) -> bool:
    r"""Validate if an input has the right shape.

    e.g. to check if an input is channel first.
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
    shape: Union[Tuple[int, ...], torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch: Optional[bool] = False,
) -> torch.Tensor:
    r"""Draw a reparameterized sample of ``shape`` from ``dist``, one draw repeated over the batch if same_on_batch.

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


def _truncate_to_start(draw: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    r"""Turn a ``[0, 1)`` draw into an integer start in ``[0, extent)`` for a crop or patch.

    The product and the floor are taken in the wider of the two dtypes and the result is cast to the dtype of
    ``extent``. Casting the draw to a narrower ``extent`` dtype first rounds a draw close to 1 up to 1.0, which
    puts the start one past the last valid position (#5052).
    """
    dtype = torch.promote_types(draw.dtype, extent.dtype)
    return (draw.to(device=extent.device, dtype=dtype) * extent.to(dtype)).floor().to(extent.dtype)


def _adapted_sampling(
    shape: Union[Tuple[int, ...], torch.Size],
    dist: torch.distributions.Distribution,
    same_on_batch: Optional[bool] = False,
) -> torch.Tensor:
    r"""Draw a sample of ``shape`` from ``dist`` without reparameterization, one draw for the batch if same_on_batch.

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
    low: Union[float, torch.Tensor],
    high: Union[float, torch.Tensor],
    same_on_batch: bool = False,
) -> torch.Tensor:
    r"""Draw a reparameterized sample of ``shape`` from Uniform(low, high), one draw for the batch if same_on_batch.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If low/high is a torch.Tensor,
    sampling will happen in the same device/dtype as low/high torch.Tensor.
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
    shape: Union[Tuple[int, ...], torch.Size],
    a: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    same_on_batch: bool = False,
) -> torch.Tensor:
    r"""Sample from a beta sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]). By default,
    same_on_batch is set to False.

    By default, sampling happens on the default device and dtype. If a/b is a torch.Tensor,
    sampling will happen in the same device/dtype as a/b torch.Tensor.
    """
    device, dtype = _extract_device_dtype(
        [a if isinstance(a, torch.Tensor) else None, b if isinstance(b, torch.Tensor) else None]
    )
    a = torch.as_tensor(a, device=device, dtype=dtype)
    b = torch.as_tensor(b, device=device, dtype=dtype)
    dist = Beta(a, b, validate_args=False)
    return _adapted_rsampling(shape, dist, same_on_batch)


def _shape_validation(param: torch.Tensor, shape: Union[Tuple[int, ...], List[int]], name: str) -> None:
    if param.shape != torch.Size(shape):
        raise AssertionError(f"Invalid shape for {name}. Expected {shape}. Got {param.shape}")


def deepcopy_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform deep copy on any dict.

    Support torch.Tensor copying here.
    """
    out = {}
    for k, v in params.items():
        # NOTE: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol
        if isinstance(v, torch.Tensor):
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
    # Nothing to override (None or an empty kwargs dict — the common forward path): return the
    # params untouched. This skips a per-call `deepcopy_dict` (a full tensor clone of every
    # param) that only existed to decouple the copy from the override — with no override there
    # is nothing to decouple. Matches the pre-existing `None` fast path.
    if not params_override:
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


def preprocess_keypoints(input: Union[torch.Tensor, Keypoints]) -> Keypoints:
    """Preprocess input keypoints."""
    # TODO: We may allow list here.
    if isinstance(input, torch.Tensor):
        if not (len(input.shape) == 3 and input.shape[1:] == torch.Size([2])):
            raise RuntimeError(f"Only BxNx2 torch.Tensor is supported. Got {input.shape}.")
        input = Keypoints(input, False)
    if isinstance(input, Keypoints):
        raise RuntimeError(f"Expect `Keypoints` type. Got {type(input)}.")
    return input


def preprocess_classes(input: torch.Tensor) -> torch.Tensor:
    """Preprocess input class tags."""
    # TODO: We may allow list here.
    return input


class MultiprocessWrapper:
    """When used as a base class, makes the class work with the 'spawn' multiprocessing context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        args = tuple(arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}

        super().__init__(*args, **kwargs)


#: How small an image each padding mode can filter, given the kernel extent ``k`` along that axis.
#: ``filter2d`` and ``filter3d`` pad an even kernel asymmetrically -- ``(k - 1) // 2`` in front and
#: ``k // 2`` behind -- so the constraint is against the wider of the two, ``k // 2``, not against the
#: radius. ``reflect`` cannot mirror a pad as wide as the axis and ``circular`` cannot wrap one wider
#: than it; ``"valid"`` is the no-padding case, where the kernel itself must fit. ``constant`` and
#: ``replicate`` invent their padding and run on a 1-pixel axis.
_MIN_FILTERED_SIZE: Dict[str, Callable[[int], int]] = {
    "reflect": lambda extent: extent // 2 + 1,
    "circular": lambda extent: extent // 2,
    "valid": lambda extent: extent,
}

#: Names of the trailing spatial axes, read from the right: a 2D filter uses the last two.
_SPATIAL_AXIS_NAMES = ("depth", "height", "width")


def _check_filter_min_size(
    name: str,
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, ...], List[int]],
    border_type: str = "reflect",
    spatial_dims: int = 2,
) -> None:
    """Refuse an image the filter's padding cannot handle, naming the class and the shape.

    torch raises about "padding size" or "calculated padded input size" from two layers below,
    which names neither the augmentation nor the image the caller passed. ``spatial_dims`` is the
    number of trailing axes the kernel spans: 2 for an image, 3 for a volume.
    """
    size = (kernel_size,) * spatial_dims if isinstance(kernel_size, int) else tuple(kernel_size)
    smallest = _MIN_FILTERED_SIZE.get(str(border_type).lower())
    if smallest is None:
        return
    subject, unit = ("an image", "pixel") if spatial_dims == 2 else ("a volume", "voxel")
    axes = _SPATIAL_AXIS_NAMES[-spatial_dims:]
    for axis, extent, side in zip(axes, size, input.shape[-spatial_dims:]):
        minimum = smallest(int(extent))
        if side < minimum:
            raise ValueError(
                f"{name} cannot filter {subject} this small: kernel_size="
                f"{tuple(int(s) for s in size)} with border_type={border_type!r} needs at least "
                f"{minimum} {unit}(s) along {axis}, but the input is "
                f"{tuple(int(s) for s in input.shape)}."
            )
