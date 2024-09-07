from typing import Optional, Union

from kornia.core.external import numpy as np
from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort

from .utils import ONNXLoader

__all__ = ["ONNXSequential"]


class ONNXSequential:
    """ONNXSequential to chain multiple ONNX operators together.

    Args:
        *args:
            A variable number of ONNX models (either ONNX ModelProto objects or file paths).
        providers:
            A list of execution providers for ONNXRuntime (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        session_options:
            Optional ONNXRuntime session options for optimizing the session.
        io_maps:
            An optional list of list of tuples specifying input-output mappings for combining models.
            If None, we assume the default input name and output name are "input" and "output" accordingly, and
            only one input and output node for each graph.
            If not None, `io_maps[0]` shall represent the `io_map` for combining the first and second ONNX models.
        cache_dir:
            cache_dir: The directory where ONNX models are cached locally (only for downloading from HuggingFace).
                Defaults to None, which will use a default `.kornia_onnx_models` directory.

    .. code-block:: python
        # Load ops from HuggingFace repos then chain to your own model!
        model = kornia.onnx.ONNXSequential(
            "hf://operators/kornia.color.gray.RgbToGrayscale",
            "hf://operators/kornia.geometry.transform.affwarp.Resize_512x512",
            "MY_OTHER_MODEL.onnx"
        )
    """

    def __init__(
        self,
        *args: Union[onnx.ModelProto, str],  # type:ignore
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,  # type:ignore
        io_maps: Optional[list[tuple[str, str]]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.onnx_loader = ONNXLoader(cache_dir)
        self.operators = args
        self._combined_op = self._combine(io_maps)
        self._session = self.create_session()

    def _load_op(self, arg: Union[onnx.ModelProto, str]) -> onnx.ModelProto:  # type:ignore
        """Loads an ONNX model, either from a file path or use the provided ONNX ModelProto.

        Args:
            arg: Either an ONNX ModelProto object or a file path to an ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        if isinstance(arg, str):
            return self.onnx_loader.load_model(arg)
        return arg

    def _combine(self, io_maps: Optional[list[tuple[str, str]]] = None) -> onnx.ModelProto:  # type:ignore
        """Combine the provided ONNX models into a single ONNX graph. Optionally, map inputs and outputs between
        operators using the `io_map`.

        Args:
            io_maps:
                A list of list of tuples representing input-output mappings for combining the models.
                Example: [[(model1_output_name, model2_input_name)], [(model2_output_name, model3_input_name)]].

        Returns:
            onnx.ModelProto: The combined ONNX model as a single ONNX graph.

        Raises:
            ValueError: If no operators are provided for combination.
        """
        if len(self.operators) == 0:
            raise ValueError("No operators found.")

        combined_op = self._load_op(self.operators[0])
        combined_op = onnx.compose.add_prefix(combined_op, prefix=f"K{str(0).zfill(2)}-")  # type:ignore

        for i, op in enumerate(self.operators[1:]):
            next_op = onnx.compose.add_prefix(self._load_op(op), prefix=f"K{str(i + 1).zfill(2)}-")  # type:ignore
            if io_maps is None:
                io_map = [(f"K{str(i).zfill(2)}-output", f"K{str(i + 1).zfill(2)}-input")]
            else:
                io_map = [(f"K{str(i).zfill(2)}-{it[0]}", f"K{str(i + 1).zfill(2)}-{it[1]}") for it in io_maps[i]]
            combined_op = onnx.compose.merge_models(combined_op, next_op, io_map=io_map)  # type:ignore

        return combined_op

    def export(self, file_path: str) -> None:
        """Export the combined ONNX model to a file.

        Args:
            file_path: str
                The file path to export the combined ONNX model.
        """
        onnx.save(self._combined_op, file_path)  # type:ignore

    def create_session(
        self,
        providers: Optional[list[str]] = None,
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
            self._combined_op.SerializeToString(),
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

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:  # type:ignore
        """Perform inference using the combined ONNX model.

        Args:
            *inputs: Inputs to the ONNX model. The number of inputs must match the expected inputs of the session.

        Returns:
            List: The outputs from the ONNX model inference.
        """
        ort_inputs = self._session.get_inputs()
        if len(ort_inputs) != len(inputs):
            raise ValueError(f"Expected {len(ort_inputs)} for the session while only {len(inputs)} received.")

        ort_input_values = {ort_inputs[i].name: inputs[i] for i in range(len(ort_inputs))}
        outputs = self._session.run(None, ort_input_values)

        return outputs
