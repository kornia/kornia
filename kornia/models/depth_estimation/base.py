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

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

from kornia.color.gray import grayscale_to_rgb
from kornia.core.check import KORNIA_CHECK
from kornia.core.external import PILImage as Image
from kornia.models._hf_models import HFONNXComunnityModel

__all__ = ["DepthEstimation"]


class DepthEstimation(HFONNXComunnityModel):
    """Base class for depth estimation models compatible with HuggingFace and ONNX."""

    name: str = "depth_estimation"

    def __call__(self, images: Union[torch.Tensor, list[torch.Tensor]]) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Estimate the depth map of each image.

        The image is cast to float32, moved to the CPU and resized there by the ``resize`` stage, if the model has
        one. ONNX Runtime runs the graphs on a NumPy array of it. The depth map is resized back to the image size on
        the CPU in float32 with a bicubic interpolation, as transformers' ``post_process_depth_estimation`` does, then
        cast to the dtype of the image and moved to its device.

        Args:
            images: If list of RGB images. Each image is a torch.Tensor with shape :math:`(3, H, W)`.
                If torch.Tensor, a torch.Tensor with shape :math:`(B, 3, H, W)`. Any floating dtype.

        Returns:
            The depth maps at the size of the images: a tensor of shape :math:`(B, 1, H, W)` for a tensor, and a
            list of tensors of shape :math:`(1, H, W)` for a list.

        Raises:
            BaseError: If an image is not of a floating dtype.

        """
        if isinstance(images, (list, tuple)):
            return [self._estimate(image[None])[0] for image in images]
        return self._estimate(images)

    def _estimate(self, images: torch.Tensor) -> torch.Tensor:
        """Return the depth maps :math:`(B, 1, H, W)` of a batch of images :math:`(B, 3, H, W)`."""
        KORNIA_CHECK(images.is_floating_point(), f"images must be of a floating dtype, got {images.dtype}")
        # float32 is the dtype the graph takes, and the resizes run in it as well, so that the depth does not depend on
        # the input dtype. They run on the CPU, where ONNX Runtime reads its input: torch 2.5.1 has no bicubic kernel
        # on MPS, and the depth does not depend on the device either.
        inputs = images.float().cpu()
        if self.resize is not None:
            inputs = self.resize(inputs)
        depth = torch.from_numpy(super().__call__(inputs.numpy())[0]).unsqueeze(1)
        # as transformers' DPT post_process_depth_estimation resizes the predicted depth to the target size
        depth = F.interpolate(depth, size=tuple(images.shape[-2:]), mode="bicubic", align_corners=False)
        return depth.to(images.dtype).to(images.device)

    def visualize(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        depth_maps: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        output_type: str = "torch",
        depth_type: str = "relative",
        max_depth: int = 80,
    ) -> Union[torch.Tensor, list[torch.Tensor], list[Image.Image]]:
        """Draw the depth maps.

        Args:
            images: a tensor of shape :math:`(B, 3, H, W)` or a list of tensors of shape :math:`(3, H, W)`.
            depth_maps: estimated depths, as the model returns them: :math:`(B, 1, H, W)` or a list of
                :math:`(1, H, W)`. Estimated from ``images`` if not given.
            output_type: type of the output.
            depth_type: 'metric' or 'relative' depth.
            max_depth: maximum depth value. Only valid for metric depth.

        Returns:
            A list with one RGB image of shape :math:`(3, H, W)` per depth map, the depth divided by its maximum
            (``"relative"``) or by ``max_depth`` (``"metric"``), converted according to ``output_type``.

        """
        if depth_maps is None:
            depth_maps = self(images)
        output = []
        for depth_map in depth_maps:
            if depth_type == "metric":
                depth_map = depth_map / max_depth
            elif depth_type == "relative":
                depth_map = depth_map / depth_map.max()
            else:
                raise ValueError(f"Unsupported depth type `{depth_type}`.")
            output.append(grayscale_to_rgb(depth_map))

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, torch.Tensor))

    def save(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        depth_maps: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
        depth_type: str = "relative",
        max_depth: int = 80,
    ) -> None:
        """Save the images and their depth maps.

        Args:
            images: a tensor of shape :math:`(B, 3, H, W)` or a list of tensors of shape :math:`(3, H, W)`.
            depth_maps: estimated depths, as the model returns them: :math:`(B, 1, H, W)` or a list of
                :math:`(1, H, W)`. Estimated from ``images`` if not given.
            output_type: type of the output.
            depth_type: 'metric' or 'relative' depth.
            max_depth: maximum depth value. Only valid for metric depth.
            directory: where to store outputs.

        """
        outputs = self.visualize(images, depth_maps, output_type, depth_type=depth_type, max_depth=max_depth)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_depth")
