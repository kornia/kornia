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

from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.core.external import onnx
from kornia.models.base import ModelBase

__all__ = ["SuperResolution"]


# TODO: support patching -> SR -> unpatching pipeline
class SuperResolution(ModelBase):
    """SuperResolution is a module that wraps an super resolution model."""

    name: str = "super_resolution"
    input_image_size: Optional[int]
    output_image_size: Optional[int]
    pseudo_image_size: Optional[int]

    @torch.inference_mode()
    def forward(self, images: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]:
        """Forward pass of the super resolution model.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            output tensor.

        """
        output = self.pre_processor(images)
        if isinstance(
            output,
            (
                list,
                tuple,
            ),
        ):
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
        images: Union[Tensor, List[Tensor]],
        edge_maps: Optional[Union[Tensor, List[Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[Tensor, List[Tensor], List["Image.Image"]]:  # type: ignore
        """Draw the super resolution results.

        Args:
            images: input tensor.
            edge_maps: detected edges.
            output_type: type of the output.

        Returns:
            output tensor.

        """
        if edge_maps is None:
            edge_maps = self.forward(images)
        output = []
        for edge_map in edge_maps:
            output.append(edge_map)

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, Tensor))

    def save(
        self,
        images: Union[Tensor, List[Tensor]],
        edge_maps: Optional[Union[Tensor, List[Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the super resolution results.

        Args:
            images: input tensor.
            edge_maps: detected edges.
            output_type: type of the output.
            directory: where to save outputs.
            output_type: backend used to generate outputs.

        Returns:
            output tensor.

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

        return super().to_onnx(
            onnx_name,
            input_shape=[-1, 3, self.input_image_size or -1, self.input_image_size or -1],
            output_shape=[-1, 3, self.output_image_size or -1, self.output_image_size or -1],
            pseudo_shape=[1, 3, self.pseudo_image_size or 352, self.pseudo_image_size or 352],
            model=self if include_pre_and_post_processor else self.model,
            save=save,
            additional_metadata=additional_metadata,
            **kwargs,
        )
