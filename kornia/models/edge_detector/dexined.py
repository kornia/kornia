from typing import Optional, Tuple

import torch
from torch import nn

from kornia.core import rand
from kornia.filters.dexined import DexiNed
from kornia.models.edge_detector.base import EdgeDetector
from kornia.models.utils import ResizePostProcessor, ResizePreProcessor


class DexiNedBuilder:
    @staticmethod
    def build(pretrained: bool = True, image_size: Optional[int] = 352) -> EdgeDetector:
        model = DexiNed(pretrained=pretrained)
        return EdgeDetector(
            model,
            ResizePreProcessor((image_size, image_size)) if image_size is not None else nn.Identity(),
            ResizePostProcessor() if image_size is not None else nn.Identity(),
        )

    @staticmethod
    def to_onnx(
        onnx_name: Optional[str] = None,
        pretrained: bool = True,
        image_size: Optional[int] = 352,
    ) -> Tuple[str, EdgeDetector]:
        edge_detector = DexiNedBuilder.build(pretrained, image_size)
        if onnx_name is None:
            onnx_name = f"Kornia-DexiNed-{image_size}.onnx"

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
