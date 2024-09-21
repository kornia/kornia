from future import __annotations__

import os
from typing import Any, Optional

from kornia.core import ImageSequential
from kornia.onnx.utils import ONNXLoader, io_name_conversion
from kornia.onnx import ONNXSequential, load, add_metadata
from kornia.config import kornia_config
from kornia.core.external import onnx
from kornia.models.base import ModelBaseMixin
from .preprocessor import PreprocessingLoader


class ONNXComunnityModelLoader:

    def __init__(
        self, model_name: str, model_type: str = "model", cache_dir: Optional[str] = None
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        
        if cache_dir is None:
            cache_dir = kornia_config.hub_onnx_dir
        self.loader = ONNXLoader(cache_dir=os.path.join(cache_dir, self.model_name))
        self.model_url = f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/onnx/{self.model_type}.onnx"
        self.config_url = f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/preprocessor_config.json"

    def load_model(self, download: bool = True, io_name_mapping: Optional[dict[str, str]] = None, **kwargs: Any) -> "onnx.ModelProto":  # type:ignore
        onnx_model = self.loader.load_model(self.model_url, download=download, **kwargs)
        json_req = self.loader.load_config(self.config_url)

        onnx_model = self._add_metadata(onnx_model, {
            "input_size": (json_req["size"]["height"], json_req["size"]["width"])
        })

        if io_name_mapping is not None:
            onnx_model = io_name_conversion(onnx_model, io_name_mapping)

        return onnx_model

    def load_preprocessing(self,) -> ImageSequential:
        json_req = self.loader.load_config(self.config_url)
        return PreprocessingLoader.from_json(json_req)

    def _add_metadata(
        self,
        model: "onnx.ModelProto",  # type:ignore
        additional_metadata: dict[str, Any] = {}
    ) -> "onnx.ModelProto":  # type:ignore
        for key, value in additional_metadata.items():
            metadata_props = model.metadata_props.add()
            metadata_props.key = key
            metadata_props.value = str(value)
        return model


class ONNXComunnityModel(ONNXSequential, ModelBaseMixin):

    name: str = "onnx_community_model"

    def __init__(
        self,
        model: "onnx.ModelProto",  # type: ignore
        pre_processor: Optional["onnx.ModelProto"] = None,  # type: ignore
        post_processor: Optional["onnx.ModelProto"] = None,  # type: ignore
        name: Optional[str] = None,
    ) -> None:
        model_list = [] if pre_processor is None else [pre_processor]
        model_list.append(model)
        if post_processor is not None:
            model_list.append(post_processor)
        super().__init__(*model_list)
        self.name = name
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        image_size: Optional[int] = None,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: list[tuple[str, str]] = [],
        **kwargs: Any
    ) -> ONNXSequential:  # type: ignore
        """Exports a depth estimation model to ONNX format.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, image_size will be dynamic.
                If None and `include_pre_and_post_processor=False`, image_size will be
                infered from the model metadata.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}.onnx"

        if image_size is None:
            if not include_pre_and_post_processor:
                for prop in self.model.metadata_props:
                    if prop.key == "input_size":
                        image_size: tuple[int, int] = eval(prop.value)
                        break

        if include_pre_and_post_processor:
            add_metadata(self.model, additional_metadata)
            if save:
                onnx.save(self.model, onnx_name, **kwargs)
            return load(self.model)

        if save:
            self.add_metadata(additional_metadata)
            self.export(onnx_name, **kwargs)
        return self
