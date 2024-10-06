from __future__ import annotations

from typing import Any, Optional, Union

from kornia.config import kornia_config
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.core.mixin.onnx import ONNXMixin, ONNXRuntimeMixin

__all__ = ["ONNXModule", "load"]


class ONNXModule(ONNXMixin, ONNXRuntimeMixin):
    f"""ONNXModule to wrap an ONNX operator.

    Args:
        arg: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
            For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
            https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.
        providers: A list of execution providers for ONNXRuntime
            (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        session_options: Optional ONNXRuntime session options for optimizing the session.
        cache_dir: The directory where ONNX models are cached locally (only for downloading from HuggingFace).
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
        target_ir_version: The target IR version to convert to.
        target_opset_version: The target OPSET version to convert to.
    """

    def __init__(
        self,
        op: Union[onnx.ModelProto, str],  # type:ignore
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,  # type:ignore
        cache_dir: Optional[str] = None,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> None:
        self.op = self._load_op(op, cache_dir)
        if target_ir_version is not None or target_opset_version is not None:
            self.op = self._onnx_version_conversion(
                self.op, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
        session = self.create_session(providers=providers, session_options=session_options)
        self.set_session(session=session)

    def create_session(
        self, providers: list[str] | None = None, session_options: Any | None = None
    ) -> ort.InferenceSession:  # type: ignore
        return super()._create_session(self.op, providers, session_options)

    def export(self, file_path: str, **kwargs: Any) -> None:
        return super()._export(self.op, file_path, **kwargs)

    def add_metadata(self, additional_metadata: list[tuple[str, str]] = []) -> onnx.ModelProto:  # type:ignore
        return super()._add_metadata(self.op, additional_metadata)


def load(model_name: Union[onnx.ModelProto, str]) -> ONNXModule:  # type:ignore
    """Load an ONNX model from either a file path or HuggingFace.

    The loaded model is an ONNXModule object, of which you may run the model with
    the `__call__` method, with less boilerplate.

    Args:
        model_name: The name of the model to load. For Hugging Face-hosted models,
            use the format 'hf://model_name'. Valid `model_name` can be found on
            https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.
    """
    return ONNXModule(model_name)
