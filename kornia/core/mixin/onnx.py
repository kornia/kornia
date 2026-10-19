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

from __future__ import annotations

import contextlib
import io
import itertools
import os
import tempfile
from typing import (
    Any,
    ClassVar,
    Iterator,
    Optional,
    Union,
)

import torch
from torch import nn

import kornia
from kornia.core._compat import torch_version_ge
from kornia.core.exceptions import ValueCheckError
from kornia.core.external import numpy as np
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.core.utils import _warn_at_caller

# The opset ``ONNXExportMixin.to_onnx`` requests: the one the dynamo exporter (the ``torch.onnx.export`` default from
# torch 2.9) builds natively, so no version conversion runs. The legacy TorchScript exporter emits it directly as well.
_ONNX_EXPORT_OPSET = 18


def _host_array(value: Any, index: int) -> Any:
    """Return ``value`` for an ONNX Runtime session: a torch tensor on the CPU as a numpy array sharing its memory."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.device.type != "cpu":
        raise ValueError(
            f"ONNX input {index} is a torch tensor on {value.device}, but the ONNX Runtime session is run on arrays "
            "in host memory, whatever its execution provider. Move it to the CPU first, with .cpu()."
        )
    if value.layout != torch.strided:
        raise TypeError(
            f"ONNX input {index} is a {value.layout} tensor, which a numpy array cannot hold: convert it with "
            ".to_dense() first."
        )
    try:
        # resolve_conj and resolve_neg return the tensor itself unless a conjugate or negative view needs materializing
        return value.detach().resolve_conj().resolve_neg().numpy()
    except TypeError as error:  # a dtype numpy has no equivalent of: bfloat16, the float8 types, quantized types
        raise TypeError(
            f"ONNX input {index} is a {value.dtype} tensor, a dtype numpy has no equivalent of: convert it, e.g. "
            "with .float() for a float32 graph."
        ) from error


def _prefix(index: int) -> str:
    """Return the prefix that ``ONNXMixin._combine`` gives the names of the model at ``index``."""
    return f"K{str(index).zfill(2)}-"


def _name_pairs(entry: Any) -> Optional[list[tuple[Any, Any]]]:
    """Return the ``(output, input)`` pairs of one ``io_maps`` entry as a list, or None if it is not pairs.

    The entry may be any iterable of pairs (a list, a ``zip``, a set, ``dict.items()``, a generator, a numpy array),
    which is read once; a pair is any sequence of two names that is not itself a string (so a string entry, whose
    items are its characters, is not pairs either).
    """
    try:
        pairs = []
        for pair in entry:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                return None
            pairs.append((pair[0], pair[1]))
    except (TypeError, KeyError, IndexError):
        return None
    return pairs


def _io_map_pairs(io_maps: Any, joints: int, models: int) -> list[list[tuple[Any, Any]]]:
    """Return the pairs of each joint of ``io_maps``, reading ``io_maps[i]`` for joint ``i`` as the merge always did.

    Any container indexed by the joint works, a dict keyed by it too; entries past the last joint are not read.

    Raises:
        ValueCheckError: If ``io_maps`` has no entry for a joint, or an entry is not an iterable of name pairs.
    """
    pairs_per_joint = []
    for joint in range(joints):
        try:
            entry = io_maps[joint]
        except (IndexError, KeyError):
            raise ValueCheckError(
                f"io_maps needs one list of (output, input) name pairs per joint, {joints} for {models} models, and "
                f"has no entry for joint {joint}."
            ) from None
        pairs = _name_pairs(entry)
        if pairs is None:
            raise ValueCheckError(
                f"io_maps[{joint}] must be a list of (output, input) name pairs: io_maps holds one such list per joint "
                f"between consecutive models, as in [[('output', 'input')]] for two models. Got {entry!r}."
            )
        pairs_per_joint.append(pairs)
    return pairs_per_joint


def _warn_on_joint_element_types(
    combined: onnx.ModelProto, following: onnx.ModelProto, pairs: list[tuple[Any, Any]], joint: int
) -> None:
    """Warn if an output of model ``joint`` and the input of the next model it feeds differ in tensor element type.

    ``merge_models`` drops the declared inputs it connects, so ONNX Runtime no longer checks them: the later model
    runs on the earlier one's type, which a model whose first nodes take any type (a ``Cast``, a ``Shape``) accepts,
    and any other fails with ONNX Runtime's own message. The merge goes on either way. Only tensor element types are
    compared, where both ends declare one; the declared dimensions are not, since a graph can declare fixed sizes where
    it accepts others.
    """
    outputs = {value.name: value.type for value in combined.graph.output}
    inputs = {value.name: value.type for value in following.graph.input}
    for output_name, input_name in pairs:
        output_type = outputs.get(f"{_prefix(joint)}{output_name}")
        input_type = inputs.get(f"{_prefix(joint + 1)}{input_name}")
        if output_type is None or input_type is None:  # merge_models names the missing one
            continue
        # a type that is not a tensor (a sequence, an optional), or none at all, reads as UNDEFINED (0) here
        output_elem, input_elem = output_type.tensor_type.elem_type, input_type.tensor_type.elem_type
        if output_elem and input_elem and output_elem != input_elem:
            _warn_at_caller(
                f"The output {str(output_name)!r} of model {joint} is {onnx.TensorProto.DataType.Name(output_elem)}, "
                f"but it feeds the input {str(input_name)!r} of model {joint + 1}, which is "
                f"{onnx.TensorProto.DataType.Name(input_elem)}. The merged graph does not check a later model's "
                "declared input types, so the chain runs on the earlier model's type, or fails in ONNX Runtime: the "
                "tensor element types at each joint should agree. Convert one of the models, unless the later one "
                "accepts both types."
            )


def _merge_opset_imports(model: onnx.ModelProto) -> None:
    """Keep one ``opset_import`` entry per domain, in place.

    ``onnx.compose.merge_models`` concatenates the entries of both models after checking that they agree, so a chain
    of ``n`` models lists each domain ``n`` times; the version converter then updates only one of them, and the
    converted graph no longer merges with a model of the target version. Entries that disagree, which the checker and
    ONNX Runtime accept, keep the last version, the one both of them read and :meth:`ONNXMixin._opset_version`
    returns; the version converter refuses such a model.
    """
    versions = {entry.domain: entry.version for entry in model.opset_import}
    del model.opset_import[:]
    model.opset_import.extend(onnx.helper.make_opsetid(domain, version) for domain, version in versions.items())


def _first_floating_tensor(module: Any) -> Optional[torch.Tensor]:
    """Return the first floating parameter of ``module``, else its first floating buffer, else ``None``."""
    if not isinstance(module, nn.Module):
        return None
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor.is_floating_point():
            return tensor
    return None


def _to_floating(value: Any, reference: torch.Tensor) -> Any:
    """Move the floating tensors in ``value`` (a tensor or a tuple) to the dtype and device of ``reference``."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(device=reference.device, dtype=reference.dtype)
    if isinstance(value, tuple):
        return tuple(_to_floating(item, reference) for item in value)
    return value


def _uses_dynamic_shapes(export_kwargs: dict[str, Any]) -> bool:
    """Whether ``ONNXExportMixin.to_onnx`` declares its dynamic dimensions with ``dynamic_shapes``.

    It does on the dynamo exporter from torch 2.9, which warns on ``dynamic_axes``, and uses ``dynamic_axes`` otherwise.
    """
    return bool(torch_version_ge(2, 9, 0) and export_kwargs.get("dynamo", True))


@contextlib.contextmanager
def _eval_mode(module: Any) -> Iterator[None]:
    """Put ``module`` in eval mode for the block, then restore the training flag of each of its submodules."""
    modes = [(m, m.training) for m in module.modules()] if isinstance(module, nn.Module) else []
    try:
        if isinstance(module, nn.Module):
            module.eval()
        yield
    finally:
        # The flag is set directly on purpose: ``train()`` recurses into the children, which would lose a mixed
        # state. It also skips ``train()`` overrides: check any new ``train()`` override with side effects against
        # this restore.
        for submodule, training in modes:
            submodule.training = training


# The attributes in which a module keeps what its last call stored for later reads: the output cache that ``show()`` /
# ``save()`` render; the parameters that ``inverse()`` and a replay use; an augmentation's transformation matrix, built
# or lazy, with the params dict and values it was computed from, which the containers' annotation handlers match
# against; a container's per-child matrices; and ``AugmentationSequential``'s input and mask dtypes. A test in
# ``tests/core/test_onnx_mixin.py`` fails when an augmentation writes an attribute outside its constructor that is
# neither listed here nor classified there.
_PER_CALL_STATE = (
    "_output_image",
    "_params",
    "_transform_matrix",
    "_lazy_matrix_args",
    "_transform_matrix_params",
    "_transform_matrix_params_snapshot",
    "_transform_matrices",
    "input_dtype",
    "mask_dtype",
)


def _items(value: Any) -> Any:
    """Return a shallow copy of the items of a list or dict ``value``, else ``None``."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return None


@contextlib.contextmanager
def _preserved_call_state(module: Any) -> Iterator[None]:
    """Keep the attributes in ``_PER_CALL_STATE`` of every module in ``module`` as they were before the block.

    An export runs the module on its dummy input. The TorchScript exporter does so outside ``is_exporting()``, so each
    module in the tree would store the dummy's output, parameters and matrices, and the dynamo exporter can put copies
    back in place of them. Each listed attribute gets back the object it held, and one the block stored on a module
    that had none is removed, so a class-level default shows through again. A list or dict also gets back its items:
    under a capture, where ``is_exporting()`` skips their reset, the policy containers append to
    ``_transform_matrices`` in place.
    """
    saved = (
        [
            (m, {name: (m.__dict__[name], _items(m.__dict__[name])) for name in _PER_CALL_STATE if name in m.__dict__})
            for m in module.modules()
        ]
        if isinstance(module, nn.Module)
        else []
    )
    try:
        yield
    finally:
        for submodule, state in saved:
            for name in _PER_CALL_STATE:
                if name not in state:
                    submodule.__dict__.pop(name, None)
                    continue
                value, items = state[name]
                if isinstance(value, list):
                    value[:] = items
                elif isinstance(value, dict):
                    value.clear()
                    value.update(items)
                submodule.__dict__[name] = value


@contextlib.contextmanager
def _preserved_rng() -> Iterator[None]:
    """Put back the global random state the block draws from.

    That is the CPU generator, every CUDA device's once CUDA is initialized (an augmentation moved to a device draws
    there even without parameters), and the MPS generator where MPS is available.
    """
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with contextlib.ExitStack() as stack:
        # ``fork_rng`` always restores the CPU generator; the device type picks the accelerator it also restores.
        stack.enter_context(torch.random.fork_rng(devices=cuda_devices, device_type="cuda"))
        if torch.backends.mps.is_available():
            stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
        yield


class ONNXExportMixin:
    """Mixin class that provides ONNX export functionality for objects that support it.

    Attributes:
        ONNX_EXPORTABLE:
            A flag indicating whether the object can be exported to ONNX. Default is True.
        ONNX_DEFAULT_INPUTSHAPE:
            Default input shape for the ONNX export. A list of integers where `-1` indicates
            dynamic dimensions. Default is [-1, 3, -1, -1]: a batch of 3-channel images of any
            batch size, height and width. A class whose models take another channel count declares its own.
        ONNX_DEFAULT_OUTPUTSHAPE:
            Default output shape for the ONNX export. A list of integers where `-1` indicates
            dynamic dimensions. Default is [-1, -1, -1, -1].
        ONNX_EXPORT_PSEUDO_SHAPE:
            This is used to create a dummy input tensor for the ONNX export. Default is [1, 3, 256, 256].
            It dimension shall match the ONNX_DEFAULT_INPUTSHAPE and ONNX_DEFAULT_OUTPUTSHAPE.
            Non-image dimensions are allowed.

    Note:
        - If `ONNX_EXPORTABLE` is False, indicating that the object cannot be exported to ONNX.

    """

    ONNX_EXPORTABLE: bool = True
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, 3, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[list[int]] = [1, 3, 256, 256]
    ADDITIONAL_METADATA: ClassVar[list[tuple[str, str]]] = []

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        input_shape: Optional[list[int]] = None,
        output_shape: Optional[list[int]] = None,
        pseudo_shape: Optional[list[int]] = None,
        model: Optional[nn.Module] = None,
        save: bool = True,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> onnx.ModelProto:
        """Export the current object to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            input_shape:
                The input shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_INPUTSHAPE` will be used, with its fixed entries (the channel count) taken
                from `pseudo_shape` when one of the same rank is given. Dynamic dimensions can be indicated by `-1`.
                Mark only the dimensions the model accepts at any size: on the dynamo exporter a `-1`
                on a dimension the model fixes, such as the channel count of a convolution, is an export error.
            output_shape:
                The output shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_OUTPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
                Only the legacy TorchScript exporter reads it; the dynamo exporter infers the output shape.
            pseudo_shape:
                The pseudo shape for the model as a list of integers. If None,
                `ONNX_EXPORT_PSEUDO_SHAPE` will be used. It needs the rank of `input_shape` when
                `input_shape` has a dynamic dimension, and a fixed entry of an explicit `input_shape` must
                match it.
            model:
                The model to export. If not provided, the current object will be used.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model. A key that is already present is overwritten.
            **kwargs:
                Additional keyword arguments to pass to the `torch.onnx.export` function.

        Raises:
            RuntimeError: If `ONNX_EXPORTABLE` is False.
            ValueError: If `input_shape` has a dynamic dimension and a rank other than the pseudo shape's, or
                a fixed entry of `input_shape` differs from the explicit `pseudo_shape`.
            FileNotFoundError: If `save` is True and the directory of `onnx_name` does not exist. This is
                checked before the export runs.

        Notes:
            - The model is exported in eval mode (``Dropout`` inactive, ``BatchNorm`` on its running
              statistics). The training flag of every submodule is restored afterwards, also when the export fails.
              The one opt-out is ``training=torch.onnx.TrainingMode.TRAINING``, which torch deprecates: the legacy
              TorchScript exporter (the default before torch 2.9, ``dynamo=False`` from it) then exports the
              training-mode graph. The dynamo exporter does not read ``training``.
            - The export runs the model on the dummy input, but these attributes of every module in the exported
              model keep the objects they held, with the same items, also when the export fails: the output cache
              that ``show()`` and ``save()`` read, and the parameters and transformation matrices that an
              augmentation's ``inverse()`` and ``transform_matrix`` read. Other attributes a module writes in its
              ``forward`` can hold what the run on the dummy left.
            - The global random state (the CPU generator, the CUDA devices' once CUDA is initialized, and the MPS
              generator where MPS is available) is restored after the dummy is drawn and after the export, so the
              draws that follow are the ones they would be without the export.
            - The export requests ONNX opset 18, which the dynamo exporter (the default of `torch.onnx.export`
              from torch 2.9) builds natively and the legacy TorchScript exporter emits directly. Pass
              `opset_version` to request another one.
            - A dummy input tensor is created from the input shape, with each `-1` taken from the pseudo shape and
              raised to 2 if it is smaller: `torch.export` specializes an example size of 0 or 1, which can leave a
              graph that declares the dimension dynamic but runs only at that size.
              It has the dtype and device of the model's first floating parameter or, for a model without one,
              of its first floating buffer; it is float32 on the CPU for a model with neither.
            - The dimensions marked `-1` are exported as dynamic: through `dynamic_shapes` on the dynamo exporter
              from torch 2.9, through `dynamic_axes` otherwise. A `dynamic_axes` or `dynamic_shapes` keyword
              argument replaces them.
            - The model is exported with `torch.onnx.export`, with constant folding enabled.

        """
        if not self.ONNX_EXPORTABLE:
            raise RuntimeError("This object cannot be exported to ONNX.")

        if input_shape is None:
            input_shape = self.ONNX_DEFAULT_INPUTSHAPE
            if pseudo_shape is not None and len(pseudo_shape) == len(input_shape):
                # The default's fixed entries (the channel count) follow the example the caller gave.
                input_shape = [dim if dim == -1 else size for dim, size in zip(input_shape, pseudo_shape)]
        elif pseudo_shape is not None and len(pseudo_shape) == len(input_shape):
            conflicts = [i for i, (dim, size) in enumerate(zip(input_shape, pseudo_shape)) if dim not in (-1, size)]
            if conflicts:
                raise ValueError(
                    f"input_shape {list(input_shape)} fixes dimension(s) {conflicts} to other sizes than pseudo_shape "
                    f"{list(pseudo_shape)}. Make the two agree, or mark those dimensions -1 in input_shape."
                )
        if output_shape is None:
            output_shape = self.ONNX_DEFAULT_OUTPUTSHAPE
        resolved_pseudo_shape = self.ONNX_EXPORT_PSEUDO_SHAPE if pseudo_shape is None else pseudo_shape
        if -1 in input_shape and len(input_shape) != len(resolved_pseudo_shape):
            raise ValueError(
                f"input_shape {list(input_shape)} has {len(input_shape)} dimensions, but the pseudo shape "
                f"{list(resolved_pseudo_shape)} that sizes its dynamic dimensions has {len(resolved_pseudo_shape)}. "
                "Pass a pseudo_shape of the same rank."
            )

        if onnx_name is None:
            onnx_name = f"Kornia-{self.__class__.__name__}.onnx"
        if save:
            directory = os.path.dirname(os.path.abspath(onnx_name))
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Cannot save the ONNX model to {onnx_name}: {directory} does not exist.")

        module = self if model is None else model
        if -1 in input_shape:
            # ``torch.export`` specializes an example size of 0 or 1, so a dynamic dimension traced at 1 can come out
            # fixed inside the graph while the graph input stays dynamic: on torch 2.9 DexiNed's ``Resize`` nodes kept
            # the batch of 1 and failed in ONNX Runtime at any other batch size. Dynamic dimensions are traced at 2 or
            # more.
            pseudo_shape = [
                max(size, 2) if dim == -1 else size for dim, size in zip(input_shape, resolved_pseudo_shape)
            ]
        # The dummy and the export's run on it draw from the global generators; both give back what they drew.
        with _preserved_rng():
            dummy_input = self._create_dummy_input(input_shape, pseudo_shape)
        reference = _first_floating_tensor(module)
        if reference is not None:
            dummy_input = _to_floating(dummy_input, reference)

        default_args: dict[str, Any] = {
            "export_params": True,
            "opset_version": _ONNX_EXPORT_OPSET,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
        }
        if "dynamic_axes" not in kwargs and "dynamic_shapes" not in kwargs:
            if _uses_dynamic_shapes(kwargs):
                default_args["dynamic_shapes"] = self._create_dynamic_shapes(input_shape, dummy_input)
            else:
                default_args["dynamic_axes"] = self._create_dynamic_axes(input_shape, output_shape)
        default_args.update(kwargs)

        # The dynamo exporter captures the model in whatever mode it is in, so a model left in training mode would
        # export active Dropout. Each submodule's own flag is restored, since ``train(True)`` would reset them all, and
        # so is the state each module keeps from its last call, which the exporters' run on the dummy replaces.
        # A file path rather than a buffer: exporting to ``io.BytesIO`` is deprecated from torch 2.9.
        with (
            _eval_mode(module),
            _preserved_call_state(module),
            _preserved_rng(),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            tmp_path = os.path.join(tmp_dir, "model.onnx")
            torch.onnx.export(
                module,
                dummy_input,
                tmp_path,
                **default_args,
            )
            onnx_model = onnx.load(tmp_path)

        # Later entries overwrite earlier ones, so the caller's metadata wins over the class's.
        metadata = [*self.ADDITIONAL_METADATA, *(additional_metadata or [])]
        onnx_model = kornia.onnx.utils.add_metadata(onnx_model, metadata)
        if save:
            onnx.save(onnx_model, onnx_name)
        return onnx_model

    def _create_dummy_input(
        self, input_shape: list[int], pseudo_shape: Optional[list[int]] = None
    ) -> Union[tuple[Any, ...], torch.Tensor]:
        return torch.rand(
            *[
                ((self.ONNX_EXPORT_PSEUDO_SHAPE[i] if pseudo_shape is None else pseudo_shape[i]) if dim == -1 else dim)
                for i, dim in enumerate(input_shape)
            ]
        )

    def _create_dynamic_axes(self, input_shape: list[int], output_shape: list[int]) -> dict[str, dict[int, str]]:
        return {
            "input": {i: "dim_" + str(i) for i, dim in enumerate(input_shape) if dim == -1},
            "output": {i: "dim_" + str(i) for i, dim in enumerate(output_shape) if dim == -1},
        }

    def _create_dynamic_shapes(
        self, input_shape: list[int], dummy_input: Union[tuple[Any, ...], torch.Tensor]
    ) -> tuple[Optional[dict[int, str]], ...]:
        # A named dimension is ``Dim.DYNAMIC`` to ``torch.export``, and the name is kept in the ONNX graph.
        dims = {i: "dim_" + str(i) for i, dim in enumerate(input_shape) if dim == -1} or None
        if isinstance(dummy_input, tuple):
            return (dims, *([None] * (len(dummy_input) - 1)))
        return (dims,)


class ONNXRuntimeMixin:
    """Provide methods to manage ONNX Runtime inference sessions."""

    def _create_session(
        self,
        op: onnx.ModelProto,
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ) -> ort.InferenceSession:
        """Create an optimized ONNXRuntime InferenceSession for the combined model.

        Args:
            op: Onnx operation.
            providers:
                Execution providers for ONNXRuntime (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
            session_options:
                Optional ONNXRuntime session options for session configuration and optimizations, used as given.
                If None, the session uses ``ORT_ENABLE_EXTENDED`` graph optimizations.

        Returns:
            ort.InferenceSession: The ONNXRuntime session optimized for inference.

        """
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        return ort.InferenceSession(
            op.SerializeToString(),
            sess_options=session_options,
            providers=providers or ["CPUExecutionProvider"],
        )

    def set_session(self, session: ort.InferenceSession) -> None:
        """Set a custom ONNXRuntime InferenceSession.

        Args:
            session: ort.InferenceSession
                The custom ONNXRuntime session to be set for inference.

        """
        self._session = session

    def get_session(self) -> ort.InferenceSession:
        """Get the current ONNXRuntime InferenceSession.

        Returns:
            ort.InferenceSession: The current ONNXRuntime session.

        """
        return self._session

    def as_cpu(self, **kwargs: Any) -> None:
        """Set the session to run on CPU."""
        self._session.set_providers(["CPUExecutionProvider"], provider_options=[{**kwargs}])

    def as_cuda(self, device_id: int = 0, **kwargs: Any) -> None:
        """Set the session to run on CUDA.

        We set the ONNX runtime session to use CUDAExecutionProvider.
        For other CUDAExecutionProvider configurations, or CUDA/cuDNN/ONNX version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html.

        Note:
            For using CUDA ONNXRuntime, you need to install `onnxruntime-gpu`.
            For handling different CUDA version, you may refer to
            https://github.com/microsoft/onnxruntime/issues/21769#issuecomment-2295342211.

        Args:
            device_id: Select GPU to execute.
            kwargs: Additional arguments for cuda.

        """
        self._session.set_providers(["CUDAExecutionProvider"], provider_options=[{"device_id": device_id, **kwargs}])

    def as_tensorrt(self, device_id: int = 0, **kwargs: Any) -> None:
        """Set the session to run on TensorRT.

        We set the ONNX runtime session to use TensorrtExecutionProvider.
        For other TensorrtExecutionProvider configurations, or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html.

        Args:
            device_id: select GPU to execute.
            kwargs: additional arguments from TensorRT.

        """
        self._session.set_providers(
            ["TensorrtExecutionProvider"], provider_options=[{"device_id": device_id, **kwargs}]
        )

    def as_openvino(self, device_type: str = "GPU", **kwargs: Any) -> None:
        """Set the session to run on OpenVINO.

        We set the ONNX runtime session to use OpenVINOExecutionProvider.
        For other OpenVINOExecutionProvider configurations, or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html.

        Args:
            device_type: CPU, NPU, GPU, GPU.0, GPU.1 based on the available GPUs, NPU, Any valid Hetero combination,
                Any valid Multi or Auto devices combination.
            kwargs: Additional arguments for OpenVINO.

        """
        self._session.set_providers(
            ["OpenVINOExecutionProvider"], provider_options=[{"device_type": device_type, **kwargs}]
        )

    def __call__(self, *inputs: Union[np.ndarray, torch.Tensor]) -> list[np.ndarray]:
        """Perform inference using the combined ONNX model.

        Args:
            *inputs: Inputs to the ONNX model, in the order of ``get_session().get_inputs()``: numpy arrays, or
                strided torch tensors on the CPU of a dtype numpy has (not bfloat16 or float8). A tensor is passed as
                ``tensor.detach().numpy()``, which shares its memory (a conjugate or negative view is materialized
                first). The number of inputs must match the expected inputs of the session.

        Returns:
            list: The outputs from the ONNX model inference, as numpy arrays for numpy and torch inputs alike.

        Raises:
            ValueError: If the number of inputs differs from the length of ``get_session().get_inputs()``, or an
                input is a torch tensor on another device than the CPU: the session is run on arrays in host memory,
                whatever its execution provider, so move such a tensor with ``.cpu()`` first.
            TypeError: If an input is a bfloat16 tensor, which numpy has no dtype for (convert it, e.g. with
                ``.float()``), or a sparse tensor (convert it with ``.to_dense()``).

        """
        ort_inputs = self._session.get_inputs()
        if len(inputs) != len(ort_inputs):
            raise ValueError(f"Expected {len(ort_inputs)} ONNX inputs, got {len(inputs)}.")
        ort_input_values = {
            ort_input.name: _host_array(input_value, index)
            for index, (ort_input, input_value) in enumerate(zip(ort_inputs, inputs))
        }
        return self._session.run(None, ort_input_values)


class ONNXMixin:
    """Provide functionality for exporting and loading models in ONNX format."""

    def _load_op(
        self,
        arg: Union[onnx.ModelProto, str, os.PathLike],
        cache_dir: Optional[str] = None,
    ) -> onnx.ModelProto:
        """Load an ONNX model, either from a file path or use the provided ONNX ModelProto.

        Args:
            arg: An ONNX ModelProto object; a ``str`` file path, ``hf://`` name or URL of an ONNX model; or an
                :class:`os.PathLike` file path, which always names a local file. A local path may start with ``~``.
            cache_dir: Where to read onnx objects from if stored on disk.

        Returns:
            onnx.ModelProto: The loaded ONNX model.

        Raises:
            TypeError: If ``arg`` is of another type, ``bytes`` included: ONNX Runtime reads ``bytes`` as a serialized
                model, which ``onnx.load_model_from_string`` turns into a ModelProto.

        """
        if isinstance(arg, onnx.ModelProto):
            return arg
        if isinstance(arg, (str, os.PathLike)):
            return kornia.onnx.utils.ONNXLoader.load_model(arg, cache_dir=cache_dir)
        hint = " For a serialized model, pass onnx.load_model_from_string(data)." if isinstance(arg, bytes) else ""
        raise TypeError(
            "Expected an onnx.ModelProto, a str (a file path, an 'hf://' name or a URL), or an os.PathLike file path, "
            f"got {type(arg).__name__}.{hint}"
        )

    def _load_ops(
        self,
        *args: Union[onnx.ModelProto, str, os.PathLike],
        cache_dir: Optional[str] = None,
    ) -> list[onnx.ModelProto]:
        """Load multiple ONNX models or operators and returns them as a list.

        Args:
            *args: A variable number of ONNX models, each as :meth:`_load_op` takes it: ONNX ModelProto objects, or
                file paths (``str`` or :class:`os.PathLike`). For Hugging Face-hosted models, use the format
                'hf://model_name' (a ``str``). Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.
            cache_dir: Where to read operations from if stored on disk.

        Returns:
            list[onnx.ModelProto]: The loaded ONNX models as a list of ONNX graphs.

        """
        op_list = []
        for arg in args:
            op_list.append(self._load_op(arg, cache_dir=cache_dir))
        return op_list

    def _combine(
        self,
        *args: onnx.ModelProto,
        io_maps: Optional[list[list[tuple[str, str]]]] = None,
    ) -> onnx.ModelProto:
        """Combine the provided ONNX models into a single ONNX graph.

        The names in model ``i`` are prefixed with ``K<i>-`` (``K00-``, ``K01-``, ...) in the combined graph. The
        metadata of the models are merged into one entry per key, which keeps its first position and takes the value
        of the last model that sets it, as :func:`kornia.onnx.utils.add_metadata` overwrites a key. The models passed
        in are not modified.

        Args:
            args: The ONNX models, in the order they run.
            io_maps:
                One list of ``(output, input)`` name pairs per joint: ``io_maps[i]`` connects outputs of model ``i``
                to inputs of model ``i + 1``. Example: ``[[(model1_output_name, model2_input_name)],
                [(model2_output_name, model3_input_name)]]``. Each entry may be any iterable of pairs (a ``zip``,
                ``dict.items()``, ...), and ``io_maps`` anything indexed by the joint; entries past the last joint are
                not read. If None, each joint connects ``"output"`` to ``"input"``.

        Returns:
            onnx.ModelProto: The combined ONNX model as a single ONNX graph.

        Raises:
            ValueError: If no model is given.
            ValueCheckError: If ``io_maps`` has no entry for a joint, or an entry is not an iterable of name pairs.
                It is a :exc:`ValueError` too.

        Warns:
            UserWarning: If an output and the input it feeds have different tensor element types. The merged graph
                does not check a later model's declared input types, so the chain then runs on the earlier model's
                type, or fails in ONNX Runtime.

        """
        if len(args) == 0:
            raise ValueError("No operators found.")
        joints = len(args) - 1
        pairs_per_joint = None if io_maps is None else _io_map_pairs(io_maps, joints, len(args))

        # merge_models refuses models whose metadata differ for a key, such as the "version" every kornia export
        # carries. The metadata are merged here instead, and cleared on the prefixed copies that add_prefix returns.
        metadata = {entry.key: entry.value for op in args for entry in op.metadata_props}

        combined_op = onnx.compose.add_prefix(args[0], prefix=_prefix(0))
        del combined_op.metadata_props[:]
        for i, op in enumerate(args[1:]):
            next_op = onnx.compose.add_prefix(op, prefix=_prefix(i + 1))
            del next_op.metadata_props[:]
            pairs = [("output", "input")] if pairs_per_joint is None else pairs_per_joint[i]
            _warn_on_joint_element_types(combined_op, next_op, pairs, i)
            io_map = [(f"{_prefix(i)}{output}", f"{_prefix(i + 1)}{input_}") for output, input_ in pairs]
            combined_op = onnx.compose.merge_models(combined_op, next_op, io_map=io_map)

        onnx.helper.set_model_props(combined_op, metadata)
        _merge_opset_imports(combined_op)
        return combined_op

    def _export(
        self,
        op: onnx.ModelProto,
        file_path: str,
        **kwargs: Any,
    ) -> None:
        """Export the combined ONNX model to a file.

        Args:
            op: onnx operation.
            file_path:
                The file path to export the combined ONNX model.
            kwargs: Additional arguments to save onnx model.

        """
        onnx.save(op, file_path, **kwargs)

    def _add_metadata(
        self,
        op: onnx.ModelProto,
        additional_metadata: Optional[list[tuple[str, str]]] = None,
    ) -> onnx.ModelProto:
        """Add metadata to the combined ONNX model.

        The ``source`` and ``version`` entries are always written. A key that is already present is overwritten,
        so the model keeps one entry per key, as ``onnx.checker.check_model`` requires.

        Args:
            op: onnx operation.
            additional_metadata:
                A list of tuples representing additional metadata to add to the combined ONNX model.
                Example: [("model_version", "0.1"), ("date", "20240909")].

        """
        return kornia.onnx.utils.add_metadata(op, additional_metadata)

    def _onnx_version_conversion(
        self,
        op: onnx.ModelProto,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> onnx.ModelProto:
        """Automatic conversion of the model's IR/OPSET version to the given target version.

        Args:
            op: onnx operation.
            target_ir_version: The target IR version to convert to.
            target_opset_version: The target OPSET version to convert to.

        """
        # only the targets given are compared: converting re-checks the model, so one that already has them is kept
        ir_differs = target_ir_version is not None and op.ir_version != target_ir_version
        opset_differs = target_opset_version is not None and self._opset_version(op) != target_opset_version
        if ir_differs or opset_differs:
            # Check if all ops are supported in the current IR version
            model_bytes = io.BytesIO()
            onnx.save_model(op, model_bytes)
            loaded_model = onnx.load_model_from_string(model_bytes.getvalue())
            # check_model rejects duplicate metadata keys, which files that kornia 0.9.0rc1 tagged twice hold: keep one
            # entry per key with its last value, the one ONNX Runtime reads, as _combine does
            metadata = {entry.key: entry.value for entry in loaded_model.metadata_props}
            del loaded_model.metadata_props[:]
            onnx.helper.set_model_props(loaded_model, metadata)
            _merge_opset_imports(loaded_model)  # a graph combined by an earlier release lists each domain per model
            if target_opset_version is not None:
                loaded_model = onnx.version_converter.convert_version(loaded_model, target_opset_version)
            onnx.checker.check_model(loaded_model)
            # Set the IR version if it passed the checking
            if target_ir_version is not None:
                loaded_model.ir_version = target_ir_version
            op = loaded_model
        return op

    @staticmethod
    def _opset_version(op: onnx.ModelProto) -> Optional[int]:
        """Return the version of the default (``ai.onnx``) operator set that ``op`` imports, or None.

        Of several entries, the last one is returned, the one ``onnx.checker`` and ONNX Runtime read.
        """
        versions = [entry.version for entry in op.opset_import if entry.domain in ("", "ai.onnx")]
        return versions[-1] if versions else None
