from typing import Optional

from .._hf_models import ONNXComunnityModelLoader
from .base import DepthEstimation


class DepthAnythingBuilder:
    @staticmethod
    def build(
        model_name: str = "depth-anything-v2-small", model_type: str = "model", cache_dir: Optional[str] = None
    ) -> DepthEstimation:
        """Exports a DepthAnything model to an ONNX model file.

        Args:
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            model_type:
                The type of the model to be loaded. Valid model types include:
                    - `model`
                    - `model_bnb4`
                    - `model_fp16`
                    - `model_int8`
                    - `model_q4`
                    - `model_quantized`
                    - `model_uint8`
            cache_dir:
                The directory where the model should be cached.

        Returns:
            str: The name of the output ONNX file.
        """
        if model_name not in [
            "depth-anything-v2-small",
            "depth-anything-v2-base",
            "depth-anything-v2-large",
        ]:
            raise ValueError(f"{model_name} is not a valid model name.")
        loader = ONNXComunnityModelLoader(model_name, model_type=model_type, cache_dir=cache_dir)
        onnx_model = loader.load_model(
            download=True,
            io_name_mapping={"pixel_values": "input", "predicted_depth": "output"},
        )
        preproc = loader.load_preprocessing().to_onnx(save=False)
        return DepthEstimation(onnx_model, pre_processor=preproc, name=f"{model_name}_{model_type}")
