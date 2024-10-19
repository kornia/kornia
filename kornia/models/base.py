import datetime
import logging
import os
from typing import List, Optional, Union

from kornia.core import Module, Tensor, stack
from kornia.core.external import PILImage as Image
from kornia.core.external import numpy as np
from kornia.core.mixin.onnx import ONNXExportMixin
from kornia.io import write_image
from kornia.utils.image import tensor_to_image

logger = logging.getLogger(__name__)


class ModelBaseMixin:
    name: str = "model"

    def _tensor_to_type(
        self, output: Union[Tensor, List[Tensor]], output_type: str, is_batch: bool = False
    ) -> Union[Tensor, List[Tensor], List["Image.Image"]]:  # type: ignore
        """Converts the output tensor to the desired type.

        Args:
            output: The output tensor or list of tensors.
            output_type: The desired output type. Accepted values are "torch" and "pil".
            is_batch: If True, the output is expected to be a batch of tensors.

        Returns:
            The converted output tensor or list of tensors.

        Raises:
            RuntimeError: If the output type is not supported.
        """
        if output_type == "torch":
            if is_batch and not isinstance(output, Tensor):
                return stack(output)
            elif is_batch and isinstance(output, Tensor):
                return output
            elif not is_batch and isinstance(output, Tensor):
                return list(output)
            elif not is_batch and not isinstance(output, Tensor):
                return output
            return output
        elif output_type == "pil":
            out = [Image.fromarray((tensor_to_image(out_img) * 255).astype(np.uint8)) for out_img in output]  # type: ignore
            return list(out)

        raise RuntimeError(f"Unsupported output type `{output_type}`.")

    def _save_outputs(
        self, outputs: Union[Tensor, List[Tensor]], directory: Optional[str] = None, suffix: str = ""
    ) -> None:
        """Save the output image(s) to a directory.

        Args:
            outputs: output tensor.
            directory: directory to save the images.
        """
        if directory is None:
            name = f"{self.name}_{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d%H%M%S')!s}"
            directory = os.path.join("kornia_outputs", name)

        os.makedirs(directory, exist_ok=True)
        for i, out_image in enumerate(outputs):
            write_image(
                os.path.join(directory, f"{str(i).zfill(6)}{suffix}.jpg"),
                out_image.mul(255.0).byte(),
            )
        logger.info(f"Outputs are saved in {directory}")


class ModelBase(Module, ONNXExportMixin, ModelBaseMixin):
    """This class wraps a model and performs pre-processing and post-processing."""

    def __init__(
        self, model: Module, pre_processor: Module, post_processor: Module, name: Optional[str] = None
    ) -> None:
        """Construct an Object Detector object.

        Args:
            model: an object detection model.
            pre_processor: a pre-processing module
            post_processor: a post-processing module.
        """
        super().__init__()
        self.model = model.eval()
        self.pre_processor = pre_processor.eval()
        self.post_processor = post_processor.eval()
        if name is not None:
            self.name = name
