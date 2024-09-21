from typing import Any, List, Optional, Tuple, Union

from kornia.config import kornia_config
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort

from .mixin import ONNXMixin, ONNXRuntimeMixin

__all__ = ["ONNXSequential"]


class ONNXSequential(ONNXMixin, ONNXRuntimeMixin):
    f"""ONNXSequential to chain multiple ONNX operators together.

    Args:
        *args: A variable number of ONNX models (either ONNX ModelProto objects or file paths).
            For Hugging Face-hosted models, use the format 'hf://model_name'. Valid `model_name` can be found on
            https://huggingface.co/kornia/ONNX_models. Or a URL to the ONNX model.
        providers: A list of execution providers for ONNXRuntime
            (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        session_options: Optional ONNXRuntime session options for optimizing the session.
        io_maps: An optional list of list of tuples specifying input-output mappings for combining models.
            If None, we assume the default input name and output name are "input" and "output" accordingly, and
            only one input and output node for each graph.
            If not None, `io_maps[0]` shall represent the `io_map` for combining the first and second ONNX models.
        cache_dir: The directory where ONNX models are cached locally (only for downloading from HuggingFace).
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
        auto_ir_version_conversion: If True, automatically convert the model's IR version to 9, and OPSET version to 17.
            Other versions may be pointed to by `target_ir_version` and `target_opset_version`.
        target_ir_version: The target IR version to convert to.
        target_opset_version: The target OPSET version to convert to.
    """

    def __init__(
        self,
        *args: Union["onnx.ModelProto", str],  # type:ignore
        providers: Optional[List[str]] = None,
        session_options: Optional["ort.SessionOptions"] = None,  # type:ignore
        io_maps: Optional[List[Tuple[str, str]]] = None,
        cache_dir: Optional[str] = None,
        auto_ir_version_conversion: bool = True,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> None:
        self.operators = self._load_ops(*args, cache_dir=cache_dir)
        if auto_ir_version_conversion:
            self.operators = self._auto_version_conversion(
                *self.operators, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
        self._combined_op = self._combine(*self.operators, io_maps=io_maps)
        self._session = self.create_session(self._combined_op, providers=providers, session_options=session_options)

    def _auto_version_conversion(
        self,
        *args: List["onnx.ModelProto"],  # type:ignore
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> List["onnx.ModelProto"]:  # type:ignore
        """Automatic conversion of the model's IR/OPSET version to the given target version.

        If `target_ir_version` is not provided, the model is converted to 9 by default.
        If `target_opset_version` is not provided, the model is converted to 17 by default.

        Args:
            target_ir_version: The target IR version to convert to.
            target_opset_version: The target OPSET version to convert to.
        """
        # TODO: maybe another logic for versioning.
        if target_ir_version is None:
            target_ir_version = 9
        if target_opset_version is None:
            target_opset_version = 17

        op_list = []
        for op in args:
            op = super()._onnx_version_conversion(
                op, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
            op_list.append(op)
        return op_list

    def _combine(
        self,
        *args: List["onnx.ModelProto"],  # type:ignore
        io_maps: Optional[List[Tuple[str, str]]] = None,
    ) -> "onnx.ModelProto":  # type:ignore
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

    def export(self, file_path: str, **kwargs: Any) -> None:
        return super()._export(self._combined_op, file_path, **kwargs)

    def add_metadata(self, additional_metadata: List[Tuple[str]] = []) -> "onnx.ModelProto":  # type:ignore
        return super()._add_metadata(self._combined_op, additional_metadata)
