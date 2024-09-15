from typing import Optional, Tuple

import torch
from torch import nn

from kornia.core import rand, tensor
from kornia.enhance.normalize import Normalize
from kornia.filters.dexined import DexiNed
from kornia.models.edge_detection.base import EdgeDetector
from kornia.models.utils import ResizePostProcessor, ResizePreProcessor


class DexiNedBuilder:
    """DexiNedBuilder is a class that builds a DexiNed model.

    .. code-block:: python

        image = kornia.utils.sample.get_sample_images()[0][None]
        model = DexiNedBuilder.build()
        model.save(image)

    """

    @staticmethod
    def build(model_name: str = "dexined", pretrained: bool = True, image_size: Optional[int] = 352) -> EdgeDetector:
        if model_name.lower() == "dexined":
            # Normalize then scale to [0, 255]
            norm = Normalize(mean=tensor([[0.485, 0.456, 0.406]]), std=tensor([[1. / 255.] * 3]))
            model = nn.Sequential(norm, DexiNed(pretrained=pretrained), nn.Sigmoid())
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'DexiNed'.")

        return EdgeDetector(
            model,
            ResizePreProcessor(image_size, image_size) if image_size is not None else nn.Identity(),
            ResizePostProcessor() if image_size is not None else nn.Identity(),
        )

    @staticmethod
    def to_onnx(
        model_name: str = "dexined",
        onnx_name: Optional[str] = None,
        pretrained: bool = True,
        image_size: Optional[int] = 352,
    ) -> Tuple[str, EdgeDetector]:
        edge_detector = DexiNedBuilder.build(model_name, pretrained, image_size)
        if onnx_name is None:
            onnx_name = f"kornia_{model_name.lower()}_{image_size}.onnx"

        if image_size is None:
            val_image = rand(1, 3, 352, 352)
        else:
            val_image = rand(1, 3, image_size, image_size)

        dynamic_axes = {"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}}
        torch.onnx.export(
            edge_detector,
            val_image,
            onnx_name,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        return onnx_name, edge_detector
