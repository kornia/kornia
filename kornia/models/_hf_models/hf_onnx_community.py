from __future__ import annotations

import os
from typing import Any, Optional

from kornia.config import kornia_config
from kornia.core import ImageSequential, Tensor
from kornia.core.external import onnx
from kornia.geometry.transform import resize
from kornia.models.base import ModelBaseMixin
from kornia.onnx import ONNXSequential
from kornia.onnx.utils import ONNXLoader, io_name_conversion

from .preprocessor import PreprocessingLoader


class HFONNXComunnityModelLoader:
    f"""Initializes the ONNXComunnityModelLoader for onnx-community repo of Hugging Face.

    Args:
        model_name: The name of the model to load.
        model_type: The type of the model to load.
        cache_dir: The directory where models are cached locally.
            Defaults to None, which will use a default `{kornia_config.hub_onnx_dir}` directory.
        with_data: Whether to download the model weights such as `model.onnx_data`.
    """

    def __init__(
        self, model_name: str, model_type: str = "model", cache_dir: Optional[str] = None, with_data: bool = False
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.with_data = with_data

        if cache_dir is None:
            cache_dir = kornia_config.hub_onnx_dir
        self.cache_dir = os.path.join(cache_dir, self.model_name)
        self.model_url = (
            f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/onnx/{self.model_type}.onnx"
        )
        self.config_url = (
            f"https://huggingface.co/onnx-community/{self.model_name}/resolve/main/preprocessor_config.json"
        )

    def load_model(
        self, download: bool = True, io_name_mapping: Optional[dict[str, str]] = None, **kwargs: Any
    ) -> onnx.ModelProto:  # type:ignore
        onnx_model = ONNXLoader.load_model(
            self.model_url, download=download, with_data=self.with_data, cache_dir=self.cache_dir, **kwargs
        )

        if io_name_mapping is not None:
            onnx_model = io_name_conversion(onnx_model, io_name_mapping)

        return onnx_model

    def load_preprocessing(
        self,
    ) -> ImageSequential:
        json_req = ONNXLoader.load_config(self.config_url)
        return PreprocessingLoader.from_json(json_req)

    def _add_metadata(
        self,
        model: onnx.ModelProto,  # type:ignore
        additional_metadata: dict[str, Any] = {},
    ) -> onnx.ModelProto:  # type:ignore
        for key, value in additional_metadata.items():
            metadata_props = model.metadata_props.add()
            metadata_props.key = key
            metadata_props.value = str(value)
        return model


class HFONNXComunnityModel(ONNXSequential, ModelBaseMixin):
    name: str = "onnx_community_model"

    def __init__(
        self,
        model: onnx.ModelProto,  # type: ignore
        pre_processor: Optional[onnx.ModelProto] = None,  # type: ignore
        post_processor: Optional[onnx.ModelProto] = None,  # type: ignore
        name: Optional[str] = None,
        auto_ir_version_conversion: bool = True,
        io_maps: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        model_list = [] if pre_processor is None else [pre_processor]
        model_list.append(model)
        if post_processor is not None:
            model_list.append(post_processor)
        super().__init__(*model_list, auto_ir_version_conversion=auto_ir_version_conversion, io_maps=io_maps)
        if name is not None:
            self.name = name
        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    def resize_back(self, images: Tensor, target_images: Tensor) -> Tensor:
        """Resize the input images back to the original size of target images.

        Args:
            images: The input images to be resized.
            target_images: The target images whose size is used as the reference for resizing.

        Returns:
            The resized images.
        """
        if isinstance(target_images, Tensor):
            return resize(images, target_images.shape[-2:])
        raise RuntimeError

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: list[tuple[str, str]] = [],
        **kwargs: Any,
    ) -> onnx.ModelProto:  # type:ignore
        """Exports a depth estimation model to ONNX format.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}.onnx"

        if not include_pre_and_post_processor:
            self._add_metadata(self.model, additional_metadata)
            if save:
                self._export(self.model, onnx_name, **kwargs)
            return self.model

        self._add_metadata(self._combined_op, additional_metadata)
        if save:
            self._export(self._combined_op, onnx_name, **kwargs)
        return self._combined_op
