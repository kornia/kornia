from future import __annotations__
from typing import Union, Optional

from kornia.color.gray import grayscale_to_rgb
from kornia.core import Tensor, tensor
from kornia.core.external import PILImage as Image
from .._hf_models.onnx_community import ONNXComunnityModel


class DepthEstimation(ONNXComunnityModel):
    
    name: str = "depth_estimation"

    def __call__(self, images: Tensor) -> Tensor:
        """Detect objects in a given list of images.

        Args:
            images: If list of RGB images. Each image is a Tensor with shape :math:`(3, H, W)`.
                If Tensor, a Tensor with shape :math:`(B, 3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        results = super().__call__(images.cpu().numpy())
        return tensor(results, device=images.device, dtype=images.dtype)

    def visualize(
        self,
        images: Tensor,
        depth_maps: Optional[Union[Tensor, list[Tensor]]] = None,
        output_type: str = "torch",
    ) -> Union[Tensor, list[Tensor], list[Image.Image]]:  # type: ignore
        """Draw the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        if depth_maps is None:
            depth_maps = self(images)
        output = []
        for depth_map in depth_maps:
            output.append(grayscale_to_rgb(depth_map)[0])

        return self._tensor_to_type(output, output_type, is_batch=isinstance(images, Tensor))

    def save(
        self,
        images: Tensor,
        depth_maps: Optional[Union[Tensor, list[Tensor]]] = None,
        directory: Optional[str] = None,
        output_type: str = "torch",
    ) -> None:
        """Save the segmentation results.

        Args:
            images: input tensor.
            output_type: type of the output.

        Returns:
            output tensor.
        """
        outputs = self.visualize(images, depth_maps, output_type)
        self._save_outputs(images, directory, suffix="_src")
        self._save_outputs(outputs, directory, suffix="_depth")