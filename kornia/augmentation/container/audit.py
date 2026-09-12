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

"""Opt-in diagnostics for the geometry of an executed augmentation pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import torch
from torch import Tensor, nn

from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation._2d.geometric.crop import RandomCrop
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation._3d.base import AugmentationBase3D
from kornia.constants import DataKey
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from .image import ImageSequential
from .ops import DataType
from .params import ParamItem
from .patch import PatchSequential
from .video import VideoSequential

if TYPE_CHECKING:
    from .augment import AugmentationSequential

__all__ = ["AugmentationAuditReport", "AugmentationAuditStep", "SpatialAudit"]

_T = TypeVar("_T")
_BOX_MODES = {DataKey.BBOX: "vertices_plus", DataKey.BBOX_XYXY: "xyxy_plus", DataKey.BBOX_XYWH: "xywh"}


def _snapshot(value: _T) -> _T:
    if isinstance(value, Tensor):
        return cast(_T, value.detach().clone())
    if isinstance(value, ParamItem):
        return cast(_T, ParamItem(value.name, _snapshot(value.data)))
    if isinstance(value, dict):
        return cast(_T, {key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return cast(_T, type(value)(_snapshot(item) for item in value))
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {
            "values": _json_value(value.detach().cpu().tolist()),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, ParamItem):
        return {"name": value.name, "data": _json_value(value.data)}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise TypeError(f"Audit metadata of type {type(value).__name__} cannot be serialized to JSON.")


@dataclass(frozen=True)
class AugmentationAuditStep:
    """Snapshot of one captured leaf operation, including repeated occurrences.

    ``matrix`` maps this occurrence's local input pixel coordinates to its output
    coordinates, including RandomCrop prepadding. The full pipeline mapping is
    stored separately on the report. ``None`` means that the
    operation has no supported 2D matrix, not that it is an identity transform.
    Parameters and flags are detached copies; ``occurrence`` is zero-based for
    each qualified module name. Shapes describe the image path. Flags include
    keyword overrides observed on that image call.
    """

    name: str
    occurrence: int
    module: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    params: dict[str, Any]
    flags: dict[str, Any]
    matrix: Tensor | None
    unsupported_reason: str | None


@dataclass(frozen=True)
class SpatialAudit:
    """Per-batch diagnostics for one returned box or keypoint input.

    Each point or box is one item. A box is outside if any corner is outside
    the inclusive pixel-center rectangle ``[0, width-1] x [0, height-1]``.
    Nonfinite items are counted separately, not counted as outside. Fractions
    use the total item count; empty inputs have a NaN fraction.

    Round trips inverse-map the *actual returned coordinates* with the recorded
    composed matrix. Errors are Euclidean point distances or symmetric Hausdorff
    distances between the four box corners (independent of corner ordering).
    Tensor box exports can lose orientation by taking an axis-aligned envelope;
    this loss is included in the measurement. ``Boxes`` objects retain corners.
    Statistics cover only finite errors, in pixels; a batch with no valid errors
    has NaN statistics and zero ``roundtrip_valid_count``. Diagnostics use float32
    for half inputs, otherwise the image dtype.
    """

    input_index: int
    data_key: str
    metric: Literal["euclidean", "corner_hausdorff"]
    count: Tensor
    out_of_frame: Tensor
    nonfinite: Tensor
    out_of_frame_fraction: Tensor
    roundtrip_valid_count: Tensor
    roundtrip_mean: Tensor
    roundtrip_median: Tensor
    roundtrip_max: Tensor


@dataclass(frozen=True)
class AugmentationAuditReport:
    """Geometry provenance and spatial diagnostics from one augmentation call.

    ``params`` is a detached snapshot of native ``ParamItem`` values for replay
    with the same pipeline configuration. JSON is a diagnostic export, not a
    parameter deserializer. Reports retain neither image pixels nor autograd
    graphs. Tensor fields are independent snapshots, not read-only tensors.

    ``geometry_status`` is ``unsupported`` when any executed operation lacks a
    supported matrix, ``singular`` when any composed matrix cannot be inverted,
    or ``available``. Inspect the per-batch ``invertible`` mask and spatial
    errors rather than interpreting availability as alignment success.
    ``inverse_matrix`` has NaNs for singular/nonfinite batches. No image inverse
    is evaluated: coordinate invertibility does not imply pixel reconstruction,
    particularly after cropping, interpolation or intensity changes.
    """

    inputs: list[dict[str, Any]]
    output_shape: tuple[int, ...]
    steps: list[AugmentationAuditStep]
    params: list[ParamItem]
    configured_extra_args: dict[DataKey, dict[str, Any]]
    geometry_status: Literal["available", "unsupported", "singular"]
    matrix: Tensor | None
    inverse_matrix: Tensor | None
    invertible: Tensor
    spatial: list[SpatialAudit]
    warnings: list[str]
    roundtrip_tolerance: float
    out_of_frame_tolerance: float
    image_reconstruction_evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export JSON-safe metadata; tensor values include dtype/device/shape, and NaNs become None."""
        return cast(dict[str, Any], _json_value(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report as strict JSON (nonfinite diagnostic values become null)."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def summary(self) -> str:
        """Describe matrix availability and diagnostic warnings without claiming image invertibility."""
        return (
            f"Augmentation audit: {len(self.steps)} captured operations, geometry {self.geometry_status}, "
            f"{len(self.spatial)} spatial inputs. Image reconstruction not evaluated."
            + ("\n" + "\n".join(self.warnings) if self.warnings else "")
        )


def _supported_sequence(module: nn.Module) -> bool:
    # Import at call time: augment imports the report types from this module.
    from .augment import AugmentationSequential

    return type(module) in (ImageSequential, AugmentationSequential)


def _leaves(sequence: ImageSequential, prefix: str = "", seen: set[int] | None = None) -> list[tuple[str, nn.Module]]:
    seen = set() if seen is None else seen
    result: list[tuple[str, nn.Module]] = []
    for name, module in sequence._modules.items():
        if module is None:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if id(module) in seen:
            raise ValueError("audit does not support a module registered under multiple names.")
        seen.add(id(module))
        if isinstance(module, (VideoSequential, PatchSequential, AugmentationBase3D)):
            raise ValueError("audit supports 2D image pipelines, not video, patch or 3D augmentations.")
        if _supported_sequence(module):
            result.extend(_leaves(cast(ImageSequential, module), path, seen))
        else:
            result.append((path, module))
    return result


def _selected_paths(sequence: ImageSequential, params: list[ParamItem], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for param in params:
        module = sequence.get_submodule(param.name)
        path = f"{prefix}.{param.name}" if prefix else param.name
        if _supported_sequence(module) and isinstance(param.data, list):
            # Use this occurrence's tree, not the nested module's last cached params.
            paths.extend(_selected_paths(cast(ImageSequential, module), param.data, path))
        else:
            paths.append(path)
    return paths


def _capture_warnings(sequence: AugmentationSequential, steps: list[AugmentationAuditStep]) -> list[str]:
    if not _supported_sequence(sequence) or sequence._params is None:
        return ["Forward capture cannot be verified for a custom container or a call without recorded parameters."]
    selected = _selected_paths(sequence, sequence._params)
    captured = [step.name for step in steps]
    if selected != captured:
        return [f"Forward capture is incomplete or out of order: selected {selected}; captured {captured}."]
    return []


def _crop_matrix(
    module: RandomCrop,
    matrix: Tensor,
    params: dict[str, Any],
    flags: dict[str, Any],
    image: Tensor,
    output: Tensor,
) -> tuple[Tensor | None, str | None]:
    applied = params["batch_prob"] > 0.5
    static = module.p == 1.0 and module.p_batch == 1.0
    changed_shape = output.shape[-2:] != image.shape[-2:]
    if not static and changed_shape and flags["cropping_mode"] == "slice" and not bool(applied.all()):
        # The shape-changing blend returns every transformed row, but slice uses
        # src indices even where the cached per-row matrix is the identity.
        return None, "mixed-application slice crop has no reliable per-row image matrix after changing shape"
    if static or changed_shape:
        # These paths return the entire transformed branch, including prepadding.
        applied = torch.ones_like(applied)
    padding = params["padding_size"].to(matrix)
    translation = torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand_as(matrix).clone()
    translation[:, 0, 2] = padding[:, 0] * applied.to(matrix)
    translation[:, 1, 2] = padding[:, 2] * applied.to(matrix)
    return matrix @ translation, None


def _coordinates(value: DataType, key: DataKey) -> Tensor:
    if (isinstance(value, Boxes) and key not in _BOX_MODES) or (
        isinstance(value, Keypoints) and key != DataKey.KEYPOINTS
    ):
        raise ValueError("audit Boxes/Keypoints objects must match their data key.")
    if isinstance(value, (Boxes, Keypoints)):
        if value._N is not None:
            raise ValueError("audit does not support ragged Boxes or Keypoints constructed from lists.")
        return value.data
    if not isinstance(value, Tensor):
        raise ValueError("audit requires batched tensors, Boxes or Keypoints; ragged inputs are unsupported.")
    if key in _BOX_MODES:
        # Match the container's public representation, including axis-aligned tensor exports.
        return Boxes.from_tensor(value, mode=_BOX_MODES[key], validate_boxes=False).data
    return value


def _capture(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
    *,
    name: str,
    steps: list[AugmentationAuditStep],
) -> None:
    image = inputs[0]
    params = _snapshot(getattr(module, "_params", {}))
    # Augmentation forward merges its kwargs into flags without necessarily
    # saving them on module.flags. `params` is a separate bound argument.
    flags = _snapshot(
        {**getattr(module, "flags", {}), **{key: value for key, value in kwargs.items() if key != "params"}}
    )
    matrix = None
    reason = None
    if isinstance(module, (GeometricAugmentationBase2D, IntensityAugmentationBase2D)):
        # Access before detaching: lazy matrix materialization must preserve ordinary autograd behavior.
        matrix = _snapshot(module.transform_matrix)
        if matrix is not None and isinstance(module, RandomCrop):
            matrix, reason = _crop_matrix(module, matrix, params, flags, image, output)
        if matrix is not None and matrix.shape != (image.shape[0], 3, 3):
            matrix = None
            reason = "operation produced a matrix with an unsupported shape"
        elif matrix is None and reason is None:
            reason = "operation did not produce a 2D matrix"
    elif type(module) is nn.Identity:
        matrix = torch.eye(3, device=image.device, dtype=image.dtype).expand(image.shape[0], -1, -1).clone()
    else:
        reason = "non-rigid or unknown operation; coordinate propagation is not certified by a 2D matrix"
    steps.append(
        AugmentationAuditStep(
            name,
            sum(step.name == name for step in steps),
            type(module).__name__,
            tuple(image.shape),
            tuple(output.shape),
            params,
            flags,
            matrix,
            reason,
        )
    )


def _matrices(
    steps: list[AugmentationAuditStep],
    image: Tensor,
    *,
    capture_complete: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    if not capture_complete:
        return None, None, torch.zeros(image.shape[0], device=image.device, dtype=torch.bool)
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    identity = torch.eye(3, device=image.device, dtype=dtype).expand(image.shape[0], -1, -1)
    matrix = identity.clone()
    for step in steps:
        if step.matrix is None or step.matrix.shape != matrix.shape:
            return None, None, torch.zeros(image.shape[0], device=image.device, dtype=torch.bool)
        matrix = step.matrix.to(matrix) @ matrix
    finite = matrix.isfinite().all(dim=(-2, -1))
    inverse, info = torch.linalg.inv_ex(torch.where(finite[:, None, None], matrix, identity))
    valid = finite & (info == 0) & inverse.isfinite().all(dim=(-2, -1))
    inverse = torch.where(valid[:, None, None], inverse, torch.full_like(inverse, float("nan")))
    return matrix, inverse, valid


def _spatial_audit(
    source: Tensor,
    result: Tensor,
    key: DataKey,
    index: int,
    shape: tuple[int, ...],
    inverse: Tensor | None,
    dtype: torch.dtype,
) -> SpatialAudit:
    if source.shape != result.shape:
        raise ValueError("audit requires spatial outputs to preserve the source shape and label cardinality.")
    source, result = source.to(dtype=dtype), result.detach().to(dtype=dtype)
    batch, count = result.shape[:2]
    finite = result.isfinite().flatten(2).all(-1)
    outside = (
        (result[..., 0] < 0)
        | (result[..., 0] > shape[-1] - 1)
        | (result[..., 1] < 0)
        | (result[..., 1] > shape[-2] - 1)
    )
    if key in _BOX_MODES:
        outside = outside.any(-1)
    counts = torch.full((batch,), count, device=result.device, dtype=torch.long)
    errors = torch.full((batch, count), float("nan"), device=result.device, dtype=dtype)
    if inverse is not None and count:
        flat = result.reshape(batch, -1, 2)
        homogeneous = torch.cat((flat, torch.ones_like(flat[..., :1])), dim=-1) @ inverse.transpose(-1, -2)
        # Exact division exposes points at projective infinity instead of replacing the divisor by one.
        restored = (homogeneous[..., :2] / homogeneous[..., 2:]).reshape_as(result)
        if key in _BOX_MODES:
            distances = torch.cdist(restored, source)
            errors = torch.maximum(distances.amin(-1).amax(-1), distances.amin(-2).amax(-1))
        else:
            errors = torch.linalg.vector_norm(restored - source, dim=-1)
    valid = errors.isfinite()
    errors = torch.where(valid, errors, torch.full_like(errors, float("nan")))
    valid_count = valid.sum(-1)
    mean = errors.nansum(-1) / valid_count
    if count:
        median = errors.nanquantile(0.5, dim=-1)
        maximum = errors.nan_to_num(nan=-float("inf")).amax(-1)
        maximum = torch.where(valid_count > 0, maximum, torch.full_like(maximum, float("nan")))
    else:
        median, maximum = mean.clone(), mean.clone()
    out_of_frame = (outside & finite).sum(-1)
    return SpatialAudit(
        index,
        key.name,
        "corner_hausdorff" if key in _BOX_MODES else "euclidean",
        counts,
        out_of_frame,
        (~finite).sum(-1),
        out_of_frame / counts,
        valid_count,
        mean,
        median,
        maximum,
    )


def audit(
    sequence: AugmentationSequential,
    *args: DataType,
    params: list[ParamItem] | None = None,
    data_keys: list[str | int | DataKey] | None = None,
    roundtrip_tolerance: float = 1e-3,
    out_of_frame_tolerance: float = 0.0,
) -> tuple[DataType | list[DataType] | dict[str, DataType], AugmentationAuditReport]:
    if not math.isfinite(roundtrip_tolerance) or roundtrip_tolerance < 0:
        raise ValueError("roundtrip_tolerance must be finite and nonnegative.")
    if not math.isfinite(out_of_frame_tolerance) or not 0 <= out_of_frame_tolerance <= 1:
        raise ValueError("out_of_frame_tolerance must be between zero and one.")
    selected_keys = sequence.data_keys if data_keys is None else data_keys
    if selected_keys is None:
        raise ValueError("audit requires explicit positional data_keys when the pipeline data_keys is None.")
    keys = [DataKey.get(key) for key in selected_keys]
    if len(keys) != len(args) or not keys or keys[0] != DataKey.INPUT or keys.count(DataKey.INPUT) != 1:
        raise ValueError("audit requires exactly one image as the first input and matching data_keys.")
    image = args[0]
    if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[0] == 0:
        raise ValueError(
            "audit requires a nonempty BCHW image tensor; dictionaries and unbatched inputs are unsupported."
        )
    leaves = _leaves(sequence)
    sources: dict[int, Tensor] = {}
    metadata = []
    for index, (value, key) in enumerate(zip(args, keys)):
        if key not in (DataKey.INPUT, DataKey.MASK, DataKey.KEYPOINTS, *_BOX_MODES):
            raise ValueError(f"audit does not support the data key {key.name}.")
        tensor = _coordinates(value, key)
        # All box coordinates have been converted to (B, N, 4, 2).
        expected_ndim = 3 if key == DataKey.KEYPOINTS else 4
        if tensor.ndim != expected_ndim or tensor.shape[0] != image.shape[0] or tensor.device != image.device:
            raise ValueError("audit inputs must be batched with the same batch size and device as the image.")
        original = value.data if isinstance(value, (Boxes, Keypoints)) else cast(Tensor, value)
        metadata.append(
            {
                "data_key": key.name,
                "shape": tuple(original.shape),
                "dtype": original.dtype,
                "device": original.device,
            }
        )
        if key == DataKey.KEYPOINTS or key in _BOX_MODES:
            sources[index] = _snapshot(tensor)
    steps: list[AugmentationAuditStep] = []
    handles = [
        module.register_forward_hook(partial(_capture, name=name, steps=steps), with_kwargs=True)
        for name, module in leaves
    ]
    try:
        outputs = sequence(*args, params=params, data_keys=keys)
    finally:
        for handle in handles:
            handle.remove()
    values = outputs if isinstance(outputs, list) else [outputs]
    output_image = values[0]
    if not isinstance(output_image, Tensor) or output_image.ndim != 4 or output_image.shape[0] != image.shape[0]:
        raise ValueError("audit requires operations to preserve the image batch dimension and BCHW layout.")
    output_shape = tuple(output_image.shape)
    with torch.no_grad():
        capture_warnings = _capture_warnings(sequence, steps)
        matrix, inverse, valid = _matrices(steps, image, capture_complete=not capture_warnings)
        status: Literal["available", "unsupported", "singular"] = (
            "unsupported" if matrix is None else "available" if bool(valid.all()) else "singular"
        )
        diagnostic_dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        spatial = [
            _spatial_audit(
                source,
                _coordinates(values[index], keys[index]),
                keys[index],
                index,
                output_shape,
                inverse,
                diagnostic_dtype,
            )
            for index, source in sources.items()
        ]
        warnings = capture_warnings + [
            f"{step.name}: {step.unsupported_reason}." for step in steps if step.unsupported_reason
        ]
        if status == "singular":
            warnings.append("Some composed matrices are singular or nonfinite; inspect the invertible mask.")
        for step in steps:
            if "Crop" in step.module or any(
                after < before for before, after in zip(step.input_shape[-2:], step.output_shape[-2:])
            ):
                warnings.append(
                    f"{step.name}: cropping or downsampling may discard image content "
                    "even when coordinates are invertible."
                )
        for item in spatial:
            if bool((item.nonfinite > 0).any()):
                warnings.append(f"Input {item.input_index}: nonfinite output coordinates.")
            if bool((item.out_of_frame_fraction > out_of_frame_tolerance).any()):
                warnings.append(f"Input {item.input_index}: out-of-frame fraction exceeds {out_of_frame_tolerance:g}.")
            if inverse is not None and bool((item.roundtrip_valid_count < item.count).any()):
                warnings.append(
                    f"Input {item.input_index}: some round-trip errors are nonfinite; inspect valid counts."
                )
            if bool((item.roundtrip_max > roundtrip_tolerance).any()):
                warnings.append(f"Input {item.input_index}: round-trip error exceeds {roundtrip_tolerance:g} pixels.")
        report = AugmentationAuditReport(
            metadata,
            output_shape,
            steps,
            _snapshot(sequence._params or []),
            _snapshot(sequence.extra_args),
            status,
            matrix,
            inverse,
            valid,
            spatial,
            warnings,
            roundtrip_tolerance,
            out_of_frame_tolerance,
        )
    return outputs, report
