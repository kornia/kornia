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

import io
import itertools
import os
import tempfile
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
)

import torch
from torch import nn

import kornia
from kornia.core._compat import torch_version_ge
from kornia.core.external import numpy as np
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort

# The opset ``ONNXExportMixin.to_onnx`` requests: the one the dynamo exporter (the ``torch.onnx.export`` default from
# torch 2.9) builds natively, so no version conversion runs. The legacy TorchScript exporter emits it directly as well.
_ONNX_EXPORT_OPSET = 18


def _first_floating_tensor(module: Any) -> Optional[torch.Tensor]:
    """Return the first floating parameter or buffer of ``module``, or ``None``."""
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
    ) -> onnx.ModelProto:  # type: ignore
        """Export the current object to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            input_shape:
                The input shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_INPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
                Mark only the dimensions the model accepts at any size: on the dynamo exporter a `-1`
                on a dimension the model fixes, such as the channel count of a convolution, is an export error.
            output_shape:
                The output shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_OUTPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
                Only the legacy TorchScript exporter reads it; the dynamo exporter infers the output shape.
            pseudo_shape:
                The pseudo shape for the model as a list of integers. If None,
                `ONNX_EXPORT_PSEUDO_SHAPE` will be used. It needs the rank of `input_shape` when
                `input_shape` has a dynamic dimension.
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
            ValueError: If `input_shape` has a dynamic dimension and a rank other than the pseudo shape's.
            FileNotFoundError: If `save` is True and the directory of `onnx_name` does not exist. This is
                checked before the export runs.

        Notes:
            - The model is exported in eval mode (``Dropout`` inactive, ``BatchNorm`` on its running
              statistics). The training flag of every submodule is restored afterwards, also when the export fails.
            - The export requests ONNX opset 18, which the dynamo exporter (the default of `torch.onnx.export`
              from torch 2.9) builds natively and the legacy TorchScript exporter emits directly. Pass
              `opset_version` to request another one.
            - A dummy input tensor is created from the input shape, with each `-1` taken from the pseudo shape.
              It has the dtype and device of the model's first floating parameter or buffer, and is float32 on
              the CPU for a model without one.
            - The dimensions marked `-1` are exported as dynamic: through `dynamic_shapes` on the dynamo exporter
              from torch 2.9, through `dynamic_axes` otherwise. A `dynamic_axes` or `dynamic_shapes` keyword
              argument replaces them.
            - The model is exported with `torch.onnx.export`, with constant folding enabled.

        """
        if not self.ONNX_EXPORTABLE:
            raise RuntimeError("This object cannot be exported to ONNX.")

        if input_shape is None:
            input_shape = self.ONNX_DEFAULT_INPUTSHAPE
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
            # The dynamo exporter (the default from torch 2.9) takes ``dynamic_shapes`` and warns on ``dynamic_axes``.
            if torch_version_ge(2, 9, 0) and kwargs.get("dynamo", True):
                default_args["dynamic_shapes"] = self._create_dynamic_shapes(input_shape, dummy_input)
            else:
                default_args["dynamic_axes"] = self._create_dynamic_axes(input_shape, output_shape)
        default_args.update(kwargs)

        # The dynamo exporter captures the model in whatever mode it is in, so a model left in training mode would
        # export active Dropout. Each submodule's own flag is restored, since ``train(True)`` would reset them all.
        modes = [(m, m.training) for m in module.modules()] if isinstance(module, nn.Module) else []
        try:
            if isinstance(module, nn.Module):
                module.eval()
            # A file path rather than a buffer: exporting to ``io.BytesIO`` is deprecated from torch 2.9.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, "model.onnx")
                torch.onnx.export(
                    module,  # type: ignore[arg-type]
                    dummy_input,  # type: ignore[arg-type]
                    tmp_path,
                    **default_args,
                )
                onnx_model = onnx.load(tmp_path)  # type: ignore
        finally:
            for submodule, training in modes:
                submodule.training = training

        # Later entries overwrite earlier ones, so the caller's metadata wins over the class's.
        metadata = [*self.ADDITIONAL_METADATA, *(additional_metadata or [])]
        onnx_model = kornia.onnx.utils.add_metadata(onnx_model, metadata)
        if save:
            onnx.save(onnx_model, onnx_name)  # type: ignore
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
        op: onnx.ModelProto,  # type:ignore
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,  # type:ignore
    ) -> ort.InferenceSession:  # type:ignore
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
            session_options = ort.SessionOptions()  # type:ignore
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # type:ignore
        return ort.InferenceSession(  # type:ignore
            op.SerializeToString(),
            sess_options=session_options,
            providers=providers or ["CPUExecutionProvider"],
        )

    def set_session(self, session: ort.InferenceSession) -> None:  # type: ignore
        """Set a custom ONNXRuntime InferenceSession.

        Args:
            session: ort.InferenceSession
                The custom ONNXRuntime session to be set for inference.

        """
        self._session = session

    def get_session(self) -> ort.InferenceSession:  # type: ignore
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

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:  # type:ignore
        """Perform inference using the combined ONNX model.

        Args:
            *inputs: Inputs to the ONNX model. The number of inputs must match the expected inputs of the session.

        Returns:
            list: The outputs from the ONNX model inference.

        """
        ort_inputs = self._session.get_inputs()
        ort_input_values = {ort_inputs[i].name: inputs[i] for i in range(len(ort_inputs))}
        return self._session.run(None, ort_input_values)


class ONNXMixin:
    """Provide functionality for exporting and loading models in ONNX format."""

    def _load_op(
        self,
        arg: Union[onnx.ModelProto, str],  # type:ignore
        cache_dir: Optional[str] = None,
    ) -> onnx.ModelProto:  # type:ignore
        """Load an ONNX model, either from a file path or use the provided ONNX ModelProto.

        Args:
            arg: Either an ONNX ModelProto object or a file path to an ONNX model.
            cache_dir: Where to read onnx objects from if stored on disk.

        Returns:
            onnx.ModelProto: The loaded ONNX model.

        """
        if isinstance(arg, str):
            return kornia.onnx.utils.ONNXLoader.load_model(arg, cache_dir=cache_dir)
        if isinstance(arg, onnx.ModelProto):  # type:ignore
            return arg
        raise ValueError(f"Invalid argument type. Got {type(arg)}")

    def _load_ops(
        self,
        *args: Union[onnx.ModelProto, str],  # type:ignore
        cache_dir: Optional[str] = None,
    ) -> list[onnx.ModelProto]:  # type:ignore
        """Load multiple ONNX models or operators and returns them as a list.

        Args:
            *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
                For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
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
        *args: list[onnx.ModelProto],  # type:ignore
        io_maps: Optional[list[tuple[str, str]]] = None,
    ) -> onnx.ModelProto:  # type:ignore
        """Combine the provided ONNX models into a single ONNX graph.

        Optionally, map inputs and outputs between operators using the `io_map`.

        Args:
            args: list of onnx operations.
            io_maps:
                A list of list of tuples representing input-output mappings for combining the models.
                Example: [[(model1_output_name, model2_input_name)], [(model2_output_name, model3_input_name)]].

        Returns:
            onnx.ModelProto: The combined ONNX model as a single ONNX graph.

        """
        if len(args) == 0:
            raise ValueError("No operators found.")

        combined_op = args[0]
        combined_op = onnx.compose.add_prefix(combined_op, prefix=f"K{str(0).zfill(2)}-")  # type:ignore

        for i, op in enumerate(args[1:]):
            next_op = onnx.compose.add_prefix(op, prefix=f"K{str(i + 1).zfill(2)}-")  # type:ignore
            if io_maps is None:
                io_map = [(f"K{str(i).zfill(2)}-output", f"K{str(i + 1).zfill(2)}-input")]
            else:
                io_map = [(f"K{str(i).zfill(2)}-{it[0]}", f"K{str(i + 1).zfill(2)}-{it[1]}") for it in io_maps[i]]
            combined_op = onnx.compose.merge_models(combined_op, next_op, io_map=io_map)  # type:ignore

        return combined_op

    def _export(
        self,
        op: onnx.ModelProto,  # type:ignore
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
        onnx.save(op, file_path, **kwargs)  # type:ignore

    def _add_metadata(
        self,
        op: onnx.ModelProto,  # type:ignore
        additional_metadata: Optional[list[tuple[str, str]]] = None,
    ) -> onnx.ModelProto:  # type:ignore
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
        op: onnx.ModelProto,  # type:ignore
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> onnx.ModelProto:  # type:ignore
        """Automatic conversion of the model's IR/OPSET version to the given target version.

        Args:
            op: onnx operation.
            target_ir_version: The target IR version to convert to.
            target_opset_version: The target OPSET version to convert to.

        """
        if op.ir_version != target_ir_version or self._opset_version(op) != target_opset_version:
            # Check if all ops are supported in the current IR version
            model_bytes = io.BytesIO()
            onnx.save_model(op, model_bytes)  # type:ignore
            loaded_model = onnx.load_model_from_string(model_bytes.getvalue())  # type:ignore
            if target_opset_version is not None:
                loaded_model = onnx.version_converter.convert_version(  # type:ignore
                    loaded_model, target_opset_version
                )
            onnx.checker.check_model(loaded_model)  # type:ignore
            # Set the IR version if it passed the checking
            if target_ir_version is not None:
                loaded_model.ir_version = target_ir_version
            op = loaded_model
        return op

    @staticmethod
    def _opset_version(op: onnx.ModelProto) -> Optional[int]:  # type:ignore
        """Return the version of the default (``ai.onnx``) operator set that ``op`` imports, or None."""
        for entry in op.opset_import:
            if entry.domain in ("", "ai.onnx"):
                return entry.version
        return None
