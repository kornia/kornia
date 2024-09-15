from kornia.core import Module
from typing import Optional


class ModelBase(Module):
    """This class wraps a model and performs pre-processing and post-processing."""

    name: str = "model"

    def __init__(self, model: Module, pre_processor: Module, post_processor: Module, name: Optional[str] = None) -> None:
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
