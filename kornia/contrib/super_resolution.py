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

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from kornia.config import kornia_config
from kornia.core.external import PILImage as Image
from kornia.core.external import basicsr, onnx
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.models.base import ModelBase
from kornia.models.processors import OutputRangePostProcessor, ResizePreProcessor
from kornia.models.small_sr import SmallSRNetWrapper
from kornia.utils.download import CachedDownloader

__all__ = ["RRDBNetBuilder", "SmallSRBuilder", "SuperResolution"]

URLs = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


# TODO: support patching -> SR -> unpatching pipeline
class SuperResolution(ModelBase, ONNXExportMixin):
    """SuperResolution is a module that wraps an super resolution model."""

    name: str = "super_resolution"
    input_image_size: Optional[int]
    output_image_size: Optional[int]
    pseudo_image_size: Optional[int]

    @torch.inference_mode()
    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the super resolution model.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output torch.tensor.

        """
        output = self.pre_processor(images)
        if isinstance(output, list | tuple):
            images = output[0]
        else:
            images = output
        if isinstance(images, list):
            out_images = [self.model(image[None])[0] for image in images]
        else:
            out_images = self.model(images)
        return self.post_processor(out_images)

    def visualize(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[torch.Tensor, List[torch.Tensor], List["Image.Image"]]:  # type: ignore
        """Draw the super resolution results.

        Args:
            images: input torch.tensor.
            edge_maps: detected edges.
            output_type: type of the output.

        Returns:
            output torch.tensor.

        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(edge_map)

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, torch.Tensor))

    def save(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        edge_maps: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the super resolution results.

        Args:
            images: input torch.tensor.
            edge_maps: detected edges.
            output_type: type of the output.
            directory: torch.where to save outputs.
            output_type: backend used to generate outputs.

        Returns:
            output torch.tensor.

        """
        outputs = self.visualize(images, edge_maps, output_type)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_sr")

    def to_onnx(  # type: ignore[override]
        self,
        onnx_name: Optional[str] = None,
        include_pre_and_post_processor: bool = True,
        save: bool = True,
        additional_metadata: Optional[List[Tuple[str, str]]] = None,
        **kwargs: Any,
    ) -> "onnx.ModelProto":  # type: ignore
        """Export the current super resolution model to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            image_size:
                The size to which input images will be resized during preprocessing.
                If None, image_size will be dynamic. For DexiNed, recommended scale is 352.
            include_pre_and_post_processor:
                Whether to include the pre-processor and post-processor in the exported model.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            kwargs: Additional arguments for converting to onnx.

        """
        if onnx_name is None:
            onnx_name = f"kornia_{self.name}.onnx"

        return ONNXExportMixin.to_onnx(
            self,
            onnx_name,
            input_shape=[-1, 3, self.input_image_size or -1, self.input_image_size or -1],
            output_shape=[-1, 3, self.output_image_size or -1, self.output_image_size or -1],
            pseudo_shape=[1, 3, self.pseudo_image_size or 352, self.pseudo_image_size or 352],
            model=self if include_pre_and_post_processor else self.model,
            save=save,
            additional_metadata=additional_metadata,
            **kwargs,
        )


class RRDBNetBuilder:
    @staticmethod
    def build(model_name: str = "RealESRNet_x4plus", pretrained: bool = True) -> SuperResolution:
        if model_name == "RealESRGAN_x4plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRNet_x4plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRGAN_x2plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
            )
        else:
            raise ValueError(
                f"Model {model_name} not found. Please choose from "
                "'RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus'."
            )

        model_path = None
        if pretrained:
            url = URLs[model_name]
            model_path = CachedDownloader.download_to_cache(
                url, model_name, download=True, suffix=".pth", cache_dir=kornia_config.hub_onnx_dir
            )
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["params_ema"], strict=True)
        model.eval()

        return SuperResolution(
            model,
            pre_processor=nn.Identity(),
            post_processor=OutputRangePostProcessor(min_val=0.0, max_val=1.0),
            name=model_name,
        )


class SmallSRBuilder:
    @staticmethod
    def build(
        model_name: str = "small_sr", pretrained: bool = True, upscale_factor: int = 3, image_size: Optional[int] = None
    ) -> SuperResolution:
        if model_name.lower() == "small_sr":
            model = SmallSRNetWrapper(upscale_factor, pretrained=pretrained)
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'small_sr'.")

        sr = SuperResolution(
            model,
            pre_processor=ResizePreProcessor(224, 224),
            post_processor=nn.Identity(),
            name=model_name,
        )
        if image_size is None:
            sr.pseudo_image_size = 224
        else:
            sr.input_image_size = image_size
            sr.output_image_size = image_size * 3
            sr.pseudo_image_size = image_size
        return sr
