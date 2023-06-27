from __future__ import annotations

from kornia.core import Module, Tensor


class ObjectDetector(Module):
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, imgs: Tensor) -> list[Tensor]:
        """Detect objects in images.

        Args:
            imgs: RGB images. Shape :math:`(N, 3, H, W)`

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` is class id, score, and `xywh` bounding box.
        """
        return self.model(imgs)
