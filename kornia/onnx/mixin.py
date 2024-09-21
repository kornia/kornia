from __future__ import annotations

import io
from typing import (
    Any,
    List,
    Optional,
)

from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.onnx.utils import add_metadata

from .utils import ONNXLoader


class ONNXRuntimeMixin:
    def create_session(
        self,
        op: onnx.ModelProto,  # type:ignore
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,  # type:ignore
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
            device_type: CPU, NPU, GPU, GPU.0, GPU.1 based on the avaialable GPUs, NPU, Any valid Hetero combination,
                Any valid Multi or Auto devices combination.
        """
        self._session.set_providers(
            ["OpenVINOExecutionProvider"], provider_options=[{"device_type": device_type, **kwargs}]
        )

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:  # type:ignore
        """Perform inference using the combined ONNX model.

        Args:
            *inputs: Inputs to the ONNX model. The number of inputs must match the expected inputs of the session.

        Returns:
            List: The outputs from the ONNX model inference.
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
            return ONNXLoader.load_model(arg, cache_dir=cache_dir)
        if isinstance(arg, onnx.ModelProto):  # type:ignore
            return arg
        raise ValueError(f"Invalid argument type. Got {type(arg)}")

    def _load_ops(
        self,
        *args: Union[onnx.ModelProto, str],  # type:ignore
        cache_dir: Optional[str] = None,
    ) -> List[onnx.ModelProto]:  # type:ignore
        """Loads multiple ONNX models or operators and returns them as a list.

        Args:
            *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
                For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
                https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.

        Returns:
            List[onnx.ModelProto]: The loaded ONNX models as a list of ONNX graphs.
        """
        op_list = []
        for arg in args:
            op_list.append(self._load_op(arg, cache_dir=cache_dir))
        return op_list

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
        additional_metadata: List[tuple[str, str]] = [],
    ) -> onnx.ModelProto:  # type:ignore
        """Add metadata to the combined ONNX model.

        Args:
            additional_metadata:
                A list of tuples representing additional metadata to add to the combined ONNX model.
                Example: [("version", 0.1)], [("date", 20240909)].
        """
        op = add_metadata(op, additional_metadata)
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
            onnx.checker.check_model(loaded_model)
            # Set the IR version if it passed the checking
            if target_ir_version is not None:
                loaded_model.ir_version = target_ir_version
            op = loaded_model
        return op
