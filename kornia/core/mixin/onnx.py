from __future__ import annotations

import copy
import io
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
)

import torch

import kornia
from kornia.core import Module, Tensor, rand
from kornia.core.external import numpy as np
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort


class ONNXExportMixin:
    """Mixin class that provides ONNX export functionality for objects that support it.

    Attributes:
        ONNX_EXPORTABLE:
            A flag indicating whether the object can be exported to ONNX. Default is True.
        ONNX_DEFAULT_INPUTSHAPE:
            Default input shape for the ONNX export. A list of integers where `-1` indicates
            dynamic dimensions. Default is [-1, -1, -1, -1].
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
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[list[int]] = [-1, -1, -1, -1]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[list[int]] = [1, 3, 256, 256]
    ADDITIONAL_METADATA: ClassVar[list[tuple[str, str]]] = []

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        input_shape: Optional[list[int]] = None,
        output_shape: Optional[list[int]] = None,
        pseudo_shape: Optional[list[int]] = None,
        model: Optional[Module] = None,
        save: bool = True,
        additional_metadata: list[tuple[str, str]] = [],
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type: ignore
        """Exports the current object to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            input_shape:
                The input shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_INPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
            output_shape:
                The output shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_OUTPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
            pseudo_shape:
                The pseudo shape for the model as a list of integers. If None,
                `ONNX_EXPORT_PSEUDO_SHAPE` will be used.
            model:
                The model to export. If not provided, the current object will be used.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            **kwargs:
                Additional keyword arguments to pass to the `torch.onnx.export` function.

        Notes:
            - A dummy input tensor is created based on the provided or default input shape.
            - Dynamic axes for input and output tensors are configured where dimensions are marked `-1`.
            - The model is exported with `torch.onnx.export`, with constant folding enabled and opset version set to 17.
        """
        if not self.ONNX_EXPORTABLE:
            raise RuntimeError("This object cannot be exported to ONNX.")

        if input_shape is None:
            input_shape = self.ONNX_DEFAULT_INPUTSHAPE
        if output_shape is None:
            output_shape = self.ONNX_DEFAULT_OUTPUTSHAPE

        if onnx_name is None:
            onnx_name = f"Kornia-{self.__class__.__name__}.onnx"

        dummy_input = self._create_dummy_input(input_shape, pseudo_shape)
        dynamic_axes = self._create_dynamic_axes(input_shape, output_shape)

        default_args: dict[str, Any] = {
            "export_params": True,
            "opset_version": 17,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": dynamic_axes,
        }
        default_args.update(kwargs)

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            model or self,  # type: ignore
            dummy_input,
            onnx_buffer,
            **default_args,
        )
        onnx_buffer.seek(0)
        onnx_model = onnx.load(onnx_buffer)  # type: ignore

        additional_metadata = copy.deepcopy(additional_metadata)
        additional_metadata.extend(self.ADDITIONAL_METADATA)
        onnx_model = kornia.onnx.utils.add_metadata(onnx_model, additional_metadata)
        if save:
            onnx.save(onnx_model, onnx_name)  # type: ignore
        return onnx_model

    def _create_dummy_input(
        self, input_shape: list[int], pseudo_shape: Optional[list[int]] = None
    ) -> Union[tuple[Any, ...], Tensor]:
        return rand(
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


class ONNXRuntimeMixin:
    def _create_session(
        self,
        op: onnx.ModelProto,  # type:ignore
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.InferenceSession] = None,  # type:ignore
    ) -> ort.InferenceSession:  # type:ignore
        """Create an optimized ONNXRuntime InferenceSession for the combined model.

        Args:
            providers:
                Execution providers for ONNXRuntime (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
            session_options:
                Optional ONNXRuntime session options for session configuration and optimizations.

        Returns:
            ort.InferenceSession: The ONNXRuntime session optimized for inference.
        """
        if session_options is None:
            sess_options = ort.SessionOptions()  # type:ignore
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # type:ignore
        session = ort.InferenceSession(  # type:ignore
            op.SerializeToString(),
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        return session

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
        """
        self._session.set_providers(["CUDAExecutionProvider"], provider_options=[{"device_id": device_id, **kwargs}])

    def as_tensorrt(self, device_id: int = 0, **kwargs: Any) -> None:
        """Set the session to run on TensorRT.

        We set the ONNX runtime session to use TensorrtExecutionProvider.
        For other TensorrtExecutionProvider configurations, or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html.

        Args:
            device_id: select GPU to execute.
        """
        self._session.set_providers(
            ["TensorrtExecutionProvider"], provider_options=[{"device_id": device_id, **kwargs}]
        )

    def as_openvino(self, device_type: str = "GPU", **kwargs: Any) -> None:
        """Set the session to run on TensorRT.

        We set the ONNX runtime session to use OpenVINOExecutionProvider.
        For other OpenVINOExecutionProvider configurations, or CUDA/cuDNN/ONNX/TensorRT version issues,
        you may refer to https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html.

        Args:
            device_type: CPU, NPU, GPU, GPU.0, GPU.1 based on the available GPUs, NPU, Any valid Hetero combination,
                Any valid Multi or Auto devices combination.
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
        outputs = self._session.run(None, ort_input_values)

        return outputs


class ONNXMixin:
    def _load_op(
        self,
        arg: Union[onnx.ModelProto, str],  # type:ignore
        cache_dir: Optional[str] = None,
    ) -> onnx.ModelProto:  # type:ignore
        """Loads an ONNX model, either from a file path or use the provided ONNX ModelProto.

        Args:
            arg: Either an ONNX ModelProto object or a file path to an ONNX model.

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
        """Loads multiple ONNX models or operators and returns them as a list.

        Args:
            *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
                For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.

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
        """Combine the provided ONNX models into a single ONNX graph. Optionally, map inputs and outputs between
        operators using the `io_map`.

        Args:
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
            file_path:
                The file path to export the combined ONNX model.
        """
        onnx.save(op, file_path, **kwargs)  # type:ignore

    def _add_metadata(
        self,
        op: onnx.ModelProto,  # type:ignore
        additional_metadata: list[tuple[str, str]] = [],
    ) -> onnx.ModelProto:  # type:ignore
        """Add metadata to the combined ONNX model.

        Args:
            additional_metadata:
                A list of tuples representing additional metadata to add to the combined ONNX model.
                Example: [("version", 0.1)], [("date", 20240909)].
        """
        op = kornia.onnx.utils.add_metadata(op, additional_metadata)
        return op

    def _onnx_version_conversion(
        self,
        op: onnx.ModelProto,  # type:ignore
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> onnx.ModelProto:  # type:ignore
        """Automatic conversion of the model's IR/OPSET version to the given target version.

        Args:
            target_ir_version: The target IR version to convert to.
            target_opset_version: The target OPSET version to convert to.
        """
        if op.ir_version != target_ir_version or op.opset_import[0].version != target_opset_version:
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
