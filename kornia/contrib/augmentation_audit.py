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

"""Audit one augmentation pipeline call: operation order, parameters, matrices and label round trips.

:func:`audit` runs a 2D :class:`~kornia.augmentation.container.AugmentationSequential` once and returns its ordinary
outputs with a report; the ordinary forward call is unchanged and records nothing. :ref:`augmentation-audit` explains
how to read the report and lists the pipelines it supports.

Which function or class to use:

- Run a pipeline once and get its outputs plus a report: :func:`audit`.
- Read the report: :class:`AugmentationAuditReport`; one executed operation in it: :class:`AugmentationAuditStep`;
  the out-of-frame and round-trip statistics of one keypoint or box input: :class:`SpatialAudit`.
"""

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
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation._3d.base import AugmentationBase3D
from kornia.augmentation.container.image import ImageSequential
from kornia.augmentation.container.ops import DataType
from kornia.augmentation.container.params import ParamItem
from kornia.augmentation.container.patch import PatchSequential
from kornia.augmentation.container.video import VideoSequential
from kornia.constants import DataKey
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

if TYPE_CHECKING:
    from kornia.augmentation.container.augment import AugmentationSequential

__all__ = ["AugmentationAuditReport", "AugmentationAuditStep", "SpatialAudit", "audit"]

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
    """Record one executed operation of an audit: its parameters, shapes and matrix.

    ``report.steps`` of an :class:`AugmentationAuditReport` lists one step per executed leaf operation, in execution
    order. ``name`` is the qualified module name in the pipeline, ``module`` the class name, and ``occurrence`` counts
    repeated runs of one ``name`` from 0. ``params`` and ``flags`` are detached copies; ``flags`` include the keyword
    overrides of that image call. ``input_shape`` and ``output_shape`` are those of the image.

    ``matrix`` is the ``(B, 3, 3)`` map from this operation's input pixel coordinates to its output coordinates,
    including ``RandomCrop`` prepadding. ``None`` means the operation has no supported 2D matrix, not that it is an
    identity, and ``unsupported_reason`` then says why.

    Convention:
        See :ref:`augmentation-audit`.

    See Also:
        - :class:`AugmentationAuditReport`: holds the steps and the matrix composed over all of them.
        - :class:`SpatialAudit`: the label statistics, measured with the composed matrix.
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
    """Hold the per-sample out-of-frame counts and round-trip errors of one keypoint or box input.

    ``report.spatial`` of an :class:`AugmentationAuditReport` has one entry per keypoint or box input, and each of its
    tensor fields is ``(B,)``, one value per sample. ``out_of_frame_fraction`` divides by the total item count, so an
    empty input has a NaN fraction. The round-trip statistics cover only finite errors, in pixels; a sample with no
    finite error has NaN statistics and a zero ``roundtrip_valid_count``.

    Convention:
        See :ref:`augmentation-audit`.

    See Also:
        - :class:`AugmentationAuditReport`: holds these entries with the composed matrix they invert.
        - :class:`AugmentationAuditStep`: the per-operation record, without label statistics.

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
    """Hold the executed operations, composed matrix and label diagnostics of one audit call.

    ``outputs, report = audit(sequence, image, ...)`` returns it. ``report.summary()`` describes it in text,
    ``report.to_dict()`` and ``report.to_json()`` export it, and ``sequence(image, ..., params=report.params)``
    replays the call. ``matrix`` is the ``(B, 3, 3)`` map from input to output pixel coordinates composed over
    ``steps``, or ``None`` when ``geometry_status`` is ``"unsupported"``, and ``spatial`` holds one
    :class:`SpatialAudit` per keypoint or box input.

    ``params`` is a detached snapshot of native ``ParamItem`` values. JSON is a diagnostic export, not a
    parameter deserializer. Reports retain neither image pixels nor autograd
    graphs. Tensor fields are independent snapshots, not read-only tensors.

    ``geometry_status`` is ``unsupported`` when any executed operation lacks a
    supported matrix, ``singular`` when any composed matrix cannot be inverted,
    or ``available``. Inspect the per-batch ``invertible`` mask and spatial
    errors rather than interpreting availability as alignment success.
    ``inverse_matrix`` has NaNs for singular/nonfinite batches. No image inverse
    is evaluated: coordinate invertibility does not imply pixel reconstruction,
    particularly after cropping, interpolation or intensity changes.

    Convention:
        See :ref:`augmentation-audit`.

    See Also:
        - :func:`audit`: runs a pipeline and builds this report.
        - :class:`AugmentationAuditStep`: one entry of ``steps``.
        - :class:`SpatialAudit`: one entry of ``spatial``.
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
    from kornia.augmentation.container.augment import AugmentationSequential

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
        # Materialize lazy matrices in the caller's context so the ordinary spatial
        # outputs remain identical.
        matrix = _snapshot(module.transform_matrix)
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
    """Run an augmentation pipeline once and return its outputs with a geometry audit report.

    ``outputs, report = audit(sequence, image, *labels)`` runs ``sequence`` once on a nonempty ``(B, C, H, W)`` image
    followed by optional batched masks (``(B, C, H, W)`` or ``(B, H, W)``), keypoints or boxes, in the order of
    ``data_keys``. ``outputs`` has the structure, values and gradients of an ordinary forward call, and ``report`` is
    an :class:`AugmentationAuditReport` of the executed operations, their composed matrix and the round-trip errors
    of every keypoint and box input. ``params`` and ``data_keys`` have the meaning they have in
    :meth:`~kornia.augmentation.container.AugmentationSequential.forward`.

    Unknown or non-rigid operations run normally and make the composed geometry unsupported. Keypoints and boxes through
    ``RandomElasticTransform``, ``RandomThinPlateSpline`` or ``RandomFisheye`` raise ``NotImplementedError``, as in
    the sequence (`#4420 <https://github.com/kornia/kornia/issues/4420>`_). The audit runs eagerly; do not audit the
    same stateful sequence concurrently.

    Convention:
        :ref:`augmentation-audit` states the direction of ``report.matrix``, the pixel-centre rectangle of the
        out-of-frame test, the dtypes of the diagnostics and the inputs it accepts.

    Args:
        sequence: The 2D augmentation sequence to execute and inspect.
        args: Image and optional spatial inputs accepted by the sequence.
        params: Optional sampled parameters to replay.
        data_keys: Optional positional data-key override.
        roundtrip_tolerance: Nonnegative pixel threshold for round-trip warnings.
        out_of_frame_tolerance: Fraction in ``[0, 1]`` above which an input gets a warning.

    Returns:
        A pair ``(outputs, report)``.

    Raises:
        ValueError: if a tolerance is out of range, ``data_keys`` do not start with the only image or do not match
            the inputs, a data key is unsupported, the image is not a nonempty ``(B, C, H, W)`` tensor, or a label
            input is unbatched, ragged or on another device; if the pipeline holds a video, patch or 3D
            augmentation or a module registered under two names, or an operation changes the batch, the layout or
            the label cardinality.

    See Also:
        - :meth:`~kornia.augmentation.container.AugmentationSequential.forward`: the same call without a report.
        - :class:`AugmentationAuditReport`: the report this returns.
    """
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
        expected_ndims = (3,) if key == DataKey.KEYPOINTS else (3, 4) if key == DataKey.MASK else (4,)
        if tensor.ndim not in expected_ndims or tensor.shape[0] != image.shape[0] or tensor.device != image.device:
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
    # The forward above intentionally inherits caller autocast. Diagnostics do not:
    # reduced-precision matrix products can create false round-trip errors.
    with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
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
