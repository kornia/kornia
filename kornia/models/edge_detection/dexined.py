from torch import nn

from kornia.core import tensor
from kornia.enhance.normalize import Normalize
from kornia.filters.dexined import DexiNed
from kornia.models.edge_detection.base import EdgeDetector
from kornia.models.utils import ResizePostProcessor, ResizePreProcessor

__all__ = ["DexiNedBuilder"]


class DexiNedBuilder:
    """DexiNedBuilder is a class that builds a DexiNed model.

    .. code-block:: python

        images = kornia.utils.sample.get_sample_images()
        model = DexiNedBuilder.build()
        model.save(images)
    """

    @staticmethod
    def build(model_name: str = "dexined", pretrained: bool = True, image_size: int = 352) -> EdgeDetector:
        if model_name.lower() == "dexined":
            # Normalize then scale to [0, 255]
            norm = Normalize(mean=tensor([[0.485, 0.456, 0.406]]), std=tensor([[1.0 / 255.0] * 3]))
            model = nn.Sequential(norm, DexiNed(pretrained=pretrained), nn.Sigmoid())
        else:
            raise ValueError(f"Model {model_name} not found. Please choose from 'DexiNed'.")

        return EdgeDetector(
            model,
            ResizePreProcessor(image_size, image_size),
            ResizePostProcessor(),
            name="dexined",
        )
